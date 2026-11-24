#!/usr/bin/env python3
"""Floating HUD for live lectures (tkinter, no extra dependency).

Flow:
1. ``--ask-interests`` shows a small setup card: write what you care about,
   confirm consent, choose system audio / mic.
2. The HUD starts ``live_ctl.py start`` and stays always-on-top.
3. Press Enter (HUD focused) or the global F9 / Ctrl+Alt+V to write a manual
   VAR marker; the engine saves the previous/next 60 seconds.
4. The HUD polls ``status.json``; it does not call an LLM itself. Semantic
   hits come from the host judge loop (``judge_live.py``) or the openai backend.

This is a thin control surface. Capture, ASR and clipping stay in the engine.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
LIVE_CTL = Path(__file__).resolve().parent / "live_ctl.py"
ACTIVE_STATES = {"starting", "loading_asr", "recording"}
DARK = "#1f2024"
CARD = "#2a2c31"
TEXT = "#e8e8ea"
MUTED = "#9aa0a6"
ACCENT = "#8ab4f8"
GREEN = "#81c995"
RED = "#f28b82"
YELLOW = "#fdd663"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(path)


def _fmt_elapsed(seconds) -> str:
    total = max(0, int(float(seconds or 0)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _default_config() -> dict:
    return {
        "session_id": time.strftime("lecture-%Y%m%d-%H%M%S"),
        "title": "课堂实时 VAR",
        "consent": False,
        "interests": [],
        "semantic": {
            "enabled": True, "backend": "host",
            "window_seconds": 45.0, "interval_seconds": 25.0,
            "min_new_chars": 80, "min_confidence": 0.6,
        },
        "capture": {
            "source": "loopback", "screen": True, "fps": 1.0,
            "video_fps": 5.0, "video_width": 1280, "video_crf": 28,
        },
        "asr": {"model": "base", "language": "zh"},
        "trigger": {},
    }


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_interests(text: str) -> list[dict]:
    interests = []
    for index, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        question = parts[0]
        definition = parts[1] if len(parts) > 1 else ""
        if not question:
            continue
        interests.append({
            "id": f"kp{index}",
            "question": question,
            "definition": definition,
        })
    return interests


class GlobalHotkeys:
    """Register F9 / Ctrl+Alt+V system-wide on Windows."""

    def __init__(self, root: tk.Tk, events: queue.Queue):
        self.root = root
        self.events = events
        self.thread_id = 0
        self.registered = []
        self.thread: threading.Thread | None = None
        if os.name != "nt":
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
        except AttributeError:
            return
        self.thread_id = kernel32.GetCurrentThreadId()
        if user32.RegisterHotKey(None, 1, 0, 0x78):  # F9
            self.registered.append(1)
        if user32.RegisterHotKey(None, 2, 0x0002 | 0x0001, 0x56):  # Ctrl+Alt+V
            self.registered.append(2)
        if not self.registered:
            return
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312:  # WM_HOTKEY
                self.events.put("manual")
        for hotkey_id in self.registered:
            user32.UnregisterHotKey(None, hotkey_id)

    def stop(self) -> None:
        if not self.thread_id:
            return
        try:
            ctypes.windll.user32.PostThreadMessageW(
                self.thread_id, 0x0012, 0, 0  # WM_QUIT
            )
        except (AttributeError, OSError):
            pass


class Hud:
    def __init__(self, config_path: Path, out: Path, ask: bool,
                 no_start: bool, consent: bool, python: str = ""):
        self.config_path = config_path
        self.out = out
        self.no_start = no_start
        self.consent = consent
        self.python = python or sys.executable
        self.root = tk.Tk()
        self.root.title("课堂实时 VAR")
        self.root.configure(bg=DARK)
        self.root.attributes("-topmost", True)
        self.root.geometry("380x230+120+120")
        self.root.minsize(340, 200)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._drag = None
        self._hotkeys: GlobalHotkeys | None = None
        self._started = False
        self._status = {}
        self.hotkey_queue: queue.Queue = queue.Queue()

        config = _read_json(config_path) if config_path.is_file() else {}
        self.config = _merge(_default_config(), config)
        self.consent = bool(self.consent or self.config.get("consent"))
        if ask or not config_path.is_file() or not self.config.get("interests"):
            self._build_setup()
        else:
            self._start_hud()
            self._start_engine()

    # -- setup -----------------------------------------------------------

    def _build_setup(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        self.root.overrideredirect(False)
        self.root.geometry("460x430+120+120")
        frame = tk.Frame(self.root, bg=DARK, padx=18, pady=16)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame, text="先写下你关心的知识点", bg=DARK, fg=TEXT,
            font=("Microsoft YaHei UI", 14, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            frame,
            text="每行一个，格式：问题 | 讲到什么才算教（可选）\n"
                 "判定交给 LLM；只出现关键词不算命中。",
            bg=DARK, fg=MUTED, justify="left", anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", pady=(6, 10))

        self.setup_text = tk.Text(
            frame, height=9, bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Microsoft YaHei UI", 10), wrap="word",
            padx=10, pady=8,
        )
        existing = self.config.get("interests") or []
        if existing:
            text = "\n".join(
                f"{item.get('question', '')} | {item.get('definition', '')}"
                .rstrip(" |")
                for item in existing
            )
        else:
            text = (
                "老师怎么推导动态规划的状态转移方程？ | "
                "写出状态定义、转移方程、边界和计算顺序，并用例题说明\n"
                "期末考试范围和必考题型 | 明确列出章节、题型或必考/不考"
            )
        self.setup_text.insert("1.0", text)
        self.setup_text.pack(fill="both", expand=True)

        row = tk.Frame(frame, bg=DARK)
        row.pack(fill="x", pady=(10, 4))
        tk.Label(
            row, text="声音来源", bg=DARK, fg=MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="left")
        self.source_var = tk.StringVar(
            value=str((self.config.get("capture") or {}).get("source", "loopback"))
        )
        source = ttk.Combobox(
            row, textvariable=self.source_var, width=10, state="readonly",
            values=["loopback", "mic", "both"],
        )
        source.pack(side="left", padx=(8, 0))

        self.consent_var = tk.BooleanVar(value=self.consent)
        tk.Checkbutton(
            frame, text="我确认已获得录音/录屏许可",
            variable=self.consent_var, bg=DARK, fg=TEXT,
            selectcolor=CARD, activebackground=DARK, activeforeground=TEXT,
            font=("Microsoft YaHei UI", 9), anchor="w",
        ).pack(fill="x", pady=(8, 4))

        buttons = tk.Frame(frame, bg=DARK)
        buttons.pack(fill="x", pady=(8, 0))
        tk.Button(
            buttons, text="保存并开始", command=self._setup_start,
            bg=ACCENT, fg="#111", relief="flat", padx=14, pady=6,
            font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2",
        ).pack(side="right")
        tk.Button(
            buttons, text="取消", command=self.close,
            bg=CARD, fg=TEXT, relief="flat", padx=14, pady=6,
            font=("Microsoft YaHei UI", 9), cursor="hand2",
        ).pack(side="right", padx=(0, 8))

    def _setup_start(self) -> None:
        interests = _parse_interests(self.setup_text.get("1.0", "end"))
        if not interests:
            messagebox.showwarning("还没有兴趣点", "至少写一个你关心的问题。")
            return
        if not self.consent_var.get():
            messagebox.showwarning("需要确认", "请先确认已获得录音/录屏许可。")
            return
        self.config["interests"] = interests
        self.config["consent"] = True
        self.consent = True
        self.config.setdefault("capture", {})["source"] = self.source_var.get()
        self.config.setdefault("semantic", {})["enabled"] = True
        self.config["semantic"].setdefault("backend", "host")
        _atomic_json(self.config_path, self.config)
        self._start_hud()
        self._start_engine()

    # -- hud -------------------------------------------------------------

    def _start_hud(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        self.root.overrideredirect(True)
        self.root.geometry("380x230+120+120")
        card = tk.Frame(
            self.root, bg=CARD, padx=14, pady=12,
            highlightbackground="#3c4048", highlightthickness=1,
        )
        card.pack(fill="both", expand=True, padx=6, pady=6)

        header = tk.Frame(card, bg=CARD)
        header.pack(fill="x")
        header.bind("<ButtonPress-1>", self._drag_start)
        header.bind("<B1-Motion>", self._drag_move)
        self.dot = tk.Label(
            header, text="●", bg=CARD, fg=YELLOW,
            font=("Segoe UI", 12),
        )
        self.dot.pack(side="left")
        self.title = tk.Label(
            header, text="课堂 VAR", bg=CARD, fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.title.pack(side="left", padx=(6, 0))
        self.title.bind("<ButtonPress-1>", self._drag_start)
        self.title.bind("<B1-Motion>", self._drag_move)
        tk.Button(
            header, text="×", command=self.close, bg=CARD, fg=MUTED,
            relief="flat", bd=0, font=("Segoe UI", 12), cursor="hand2",
            activebackground=CARD, activeforeground=TEXT,
        ).pack(side="right")

        self.stats = tk.Label(
            card, text="准备中…", bg=CARD, fg=TEXT, anchor="w",
            font=("Microsoft YaHei UI", 10),
        )
        self.stats.pack(fill="x", pady=(8, 2))
        self.semantic_label = tk.Label(
            card, text="", bg=CARD, fg=MUTED, anchor="w",
            font=("Microsoft YaHei UI", 8),
        )
        self.semantic_label.pack(fill="x")
        self.last_text = tk.Label(
            card, text="", bg=CARD, fg=MUTED, anchor="w", justify="left",
            wraplength=330, font=("Microsoft YaHei UI", 9),
        )
        self.last_text.pack(fill="x", pady=(6, 2))
        self.open_label = tk.Label(
            card, text="", bg=CARD, fg=GREEN, anchor="w", justify="left",
            wraplength=330, font=("Microsoft YaHei UI", 9),
        )
        self.open_label.pack(fill="x")

        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(fill="x", side="bottom", pady=(8, 0))
        self.var_button = tk.Button(
            buttons, text="手动 VAR  Enter/F9", command=self.manual_var,
            bg=ACCENT, fg="#111", relief="flat", padx=10, pady=5,
            font=("Microsoft YaHei UI", 9, "bold"), cursor="hand2",
        )
        self.var_button.pack(side="left")
        tk.Button(
            buttons, text="兴趣", command=self._edit_interests,
            bg=DARK, fg=TEXT, relief="flat", padx=10, pady=5,
            font=("Microsoft YaHei UI", 9), cursor="hand2",
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            buttons, text="停止", command=self.stop_engine,
            bg=DARK, fg=RED, relief="flat", padx=10, pady=5,
            font=("Microsoft YaHei UI", 9), cursor="hand2",
        ).pack(side="right")

        self.root.bind("<Return>", lambda _event: self.manual_var())
        self.root.bind("<F9>", lambda _event: self.manual_var())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.focus_force()
        self._hotkeys = GlobalHotkeys(self.root, self.hotkey_queue)
        self._poll()

    def _drag_start(self, event) -> None:
        self._drag = (
            event.x_root, event.y_root,
            self.root.winfo_x(), self.root.winfo_y(),
        )

    def _drag_move(self, event) -> None:
        if not self._drag:
            return
        x0, y0, wx, wy = self._drag
        self.root.geometry(f"+{wx + event.x_root - x0}+{wy + event.y_root - y0}")

    def _poll(self) -> None:
        try:
            while True:
                self.hotkey_queue.get_nowait()
                self.manual_var()
        except queue.Empty:
            pass
        self._status = _read_json(self.out / "status.json")
        status = self._status
        state = status.get("state", "starting")
        color = GREEN if state in ACTIVE_STATES else (
            RED if state == "error" else YELLOW
        )
        self.dot.configure(fg=color)
        self.stats.configure(
            text=f"{_fmt_elapsed(status.get('elapsed'))}   "
                 f"命中 {status.get('hits', 0)}   "
                 f"转写 {status.get('transcript_lines', 0)} 行"
        )
        semantic = status.get("semantic") or {}
        if semantic:
            pending = semantic.get("pending", 0)
            self.semantic_label.configure(
                text=f"语义：{semantic.get('backend', 'off')} · "
                     f"待判 {pending} · 请求 {semantic.get('requests', 0)}"
            )
        last = str(status.get("last_text") or "")
        self.last_text.configure(text=("最近：" + last[-90:]) if last else "")
        open_hits = status.get("open_hits") or []
        if open_hits:
            item = open_hits[0]
            self.open_label.configure(
                text=f"正在抓取：[{item.get('interest_id')}] "
                     f"从 {item.get('span_start_text')} 起"
            )
        elif status.get("last_hit"):
            hit = status["last_hit"]
            self.open_label.configure(
                text=f"刚完成：[{hit.get('interest_id')}] "
                     f"{hit.get('span_start_text')}–{hit.get('span_end_text')}"
            )
        else:
            self.open_label.configure(text="")
        if status.get("error"):
            self.last_text.configure(text="错误：" + str(status["error"])[:120])
        self.root.after(1000, self._poll)

    # -- actions ---------------------------------------------------------

    def _start_engine(self) -> None:
        if self._started or self.no_start:
            return
        if not self.consent:
            messagebox.showwarning(
                "需要确认", "未确认录音/录屏许可，HUD 不会启动采集。"
            )
            return
        self._started = True
        command = [
            self.python, str(LIVE_CTL), "start",
            "--config", str(self.config_path.resolve()),
            "--out", str(self.out.resolve()),
            "--consent",
        ]
        try:
            result = subprocess.run(
                command, cwd=str(ROOT), capture_output=True, text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            messagebox.showerror("启动失败", str(exc))
            return
        if result.returncode != 0:
            messagebox.showerror(
                "启动失败",
                (result.stdout + "\n" + result.stderr).strip()[:800],
            )

    def manual_var(self) -> None:
        if not self.out.is_dir():
            self.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.out / "MANUAL.json", {
            "at": None, "note": "HUD 手动 VAR",
        })
        self.var_button.configure(text="已标记 ✓")
        self.root.after(
            1500, lambda: self.var_button.configure(text="手动 VAR  Enter/F9")
        )

    def stop_engine(self) -> None:
        if not messagebox.askyesno("停止课堂", "停止录音/录屏并收尾？"):
            return
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "STOP").write_text("stop", encoding="utf-8")
        self.stats.configure(text="正在收尾…")

    def _edit_interests(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("兴趣清单")
        window.configure(bg=DARK)
        window.geometry("520x420")
        window.attributes("-topmost", True)
        tk.Label(
            window, text="兴趣清单（JSON）", bg=DARK, fg=TEXT,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))
        text = tk.Text(
            window, bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Consolas", 10), wrap="none",
        )
        text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        text.insert("1.0", json.dumps(
            self.config.get("interests") or [], ensure_ascii=False, indent=2
        ))

        def save() -> None:
            try:
                interests = json.loads(text.get("1.0", "end"))
            except ValueError as exc:
                messagebox.showerror("JSON 错误", str(exc), parent=window)
                return
            if not isinstance(interests, list) or not interests:
                messagebox.showerror("格式错误", "需要一个非空列表。", parent=window)
                return
            self.config["interests"] = interests
            _atomic_json(self.config_path, self.config)
            messagebox.showinfo(
                "已保存", "停止并重新开始后生效。", parent=window
            )
            window.destroy()

        buttons = tk.Frame(window, bg=DARK)
        buttons.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(
            buttons, text="保存", command=save, bg=ACCENT, fg="#111",
            relief="flat", padx=14, pady=5,
        ).pack(side="right")
        tk.Button(
            buttons, text="取消", command=window.destroy, bg=CARD, fg=TEXT,
            relief="flat", padx=14, pady=5,
        ).pack(side="right", padx=(0, 8))

    def close(self) -> None:
        if self._hotkeys:
            self._hotkeys.stop()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="课堂实时 VAR 悬浮球。")
    parser.add_argument("--config", default="out/interests.json",
                        help="interests.json；不存在时进入兴趣设置")
    parser.add_argument("--out", default="out/lecture", help="会话输出目录")
    parser.add_argument("--ask-interests", action="store_true",
                        help="启动前先询问兴趣点")
    parser.add_argument("--consent", action="store_true",
                        help="已确认录音/录屏许可（设置页也会确认）")
    parser.add_argument("--no-start", action="store_true",
                        help="只显示 HUD，不启动引擎（附着到已有会话）")
    parser.add_argument("--python", default="",
                        help="运行采集引擎的 Python；默认当前解释器")
    args = parser.parse_args(argv)
    return Hud(
        Path(args.config).resolve(),
        Path(args.out).resolve(),
        args.ask_interests,
        args.no_start,
        args.consent,
        args.python,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())