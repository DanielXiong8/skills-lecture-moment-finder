#!/usr/bin/env python3
"""LLM semantic trigger for live lectures.

The trigger is deliberately *not* keyword based. Every judgement is made by an
LLM that reads the last N seconds of transcript and decides whether the teacher
is actually teaching an interest right now.

Two backends:

``host``
    File based. The engine writes ``judge/requests/req_0001.json``; the host
    agent (the model that runs this Skill, e.g. CodeArts) reads it, judges with
    its own model, and writes ``judge/responses/req_0001.json``. No API key and
    no local model are required.

``openai``
    Any OpenAI compatible ``/v1/chat/completions`` endpoint. Configure with
    ``LECTURE_LLM_BASE_URL``, ``LECTURE_LLM_MODEL`` and
    ``LECTURE_LLM_API_KEY`` (or the same keys in the config file).

If the backend is unavailable the engine does **not** fall back to keywords:
it reports the error and keeps only manual VAR available. That is intentional;
keyword fallback is what produced the false positives this module replaces.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from live_core import Interest, atomic_write_json

PROTOCOL = "lecture-live-judge/1"
LEVELS = {"teach", "mention", "none"}


def _clean(value, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class SemanticSettings:
    enabled: bool = False
    backend: str = "host"
    window_seconds: float = 45.0
    interval_seconds: float = 25.0
    min_new_chars: int = 80
    max_pending: int = 2
    max_interests: int = 40
    min_confidence: float = 0.6
    timeout: float = 60.0
    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: str = ""
    openai_json_mode: bool = True

    @classmethod
    def from_config(cls, config: dict, override: str = "") -> "SemanticSettings":
        raw = config.get("semantic") if isinstance(config, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        backend = (override or str(raw.get("backend", "host"))).strip().lower()
        if backend in {"off", "none", "disabled"}:
            enabled = False
            backend = "off"
        else:
            enabled = bool(raw.get("enabled", True))
        if backend not in {"host", "openai"}:
            backend = "host"
        openai_cfg = raw.get("openai") if isinstance(raw.get("openai"), dict) else {}
        return cls(
            enabled=enabled,
            backend=backend,
            window_seconds=max(10.0, _as_float(raw.get("window_seconds"), 45.0)),
            interval_seconds=max(8.0, _as_float(raw.get("interval_seconds"), 25.0)),
            min_new_chars=max(0, int(_as_float(raw.get("min_new_chars"), 80))),
            max_pending=max(1, int(_as_float(raw.get("max_pending"), 2))),
            max_interests=max(1, int(_as_float(raw.get("max_interests"), 40))),
            min_confidence=max(
                0.0, min(1.0, _as_float(raw.get("min_confidence"), 0.6))
            ),
            timeout=max(10.0, _as_float(raw.get("timeout_seconds"), 60.0)),
            openai_base_url=str(
                openai_cfg.get("base_url")
                or os.environ.get("LECTURE_LLM_BASE_URL", "")
            ).strip(),
            openai_model=str(
                openai_cfg.get("model")
                or os.environ.get("LECTURE_LLM_MODEL", "")
            ).strip(),
            openai_api_key=str(
                openai_cfg.get("api_key")
                or os.environ.get("LECTURE_LLM_API_KEY", "")
            ).strip(),
            openai_json_mode=bool(openai_cfg.get("json_mode", True)),
        )

    def public(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "window_seconds": self.window_seconds,
            "interval_seconds": self.interval_seconds,
            "min_confidence": self.min_confidence,
            "model": self.openai_model or ("host-agent" if self.backend == "host" else ""),
            "base_url": _redact_url(self.openai_base_url),
        }


def _redact_url(url: str) -> str:
    """Never expose credentials embedded in a URL."""
    if not url:
        return ""
    return re.sub(r"//[^/@]+@", "//***@", url)


def _interest_payload(interests: list[Interest], limit: int) -> list[dict]:
    payload = []
    for interest in interests[:limit]:
        item = {
            "id": interest.id,
            "question": interest.question,
        }
        definition = getattr(interest, "definition", "") or ""
        if definition:
            item["definition"] = definition
        if interest.keywords:
            # Hints only. The judge is explicitly told not to treat a keyword
            # occurrence as a hit.
            item["hints"] = interest.keywords[:12]
        payload.append(item)
    return payload


def _window_lines(lines: list[dict], start: float, end: float) -> list[dict]:
    return [
        {"ts": round(_as_float(line.get("ts")), 2),
         "end": round(_as_float(line.get("end")), 2),
         "text": _clean(line.get("text"), 800)}
        for line in lines
        if _as_float(line.get("end")) > start and _as_float(line.get("ts")) < end
    ]


def build_prompt(interests: list[Interest], lines: list[dict],
                 window_start: float, window_end: float,
                 max_interests: int = 40) -> list[dict]:
    """Build the chat messages shared by both backends."""
    interest_json = json.dumps(
        _interest_payload(interests, max_interests), ensure_ascii=False, indent=2
    )
    transcript_json = json.dumps(
        _window_lines(lines, window_start, window_end), ensure_ascii=False, indent=2
    )
    system = (
        "你是课堂实时助教，只做一件事：判断老师此刻是否正在讲解某个兴趣知识点。\n"
        "严格区分“正在教”和“只是提到”：\n"
        "- teach：正在定义、解释、推导、证明，或用完整例题说明该知识点；\n"
        "- mention：只出现名称、考试范围、复习清单，或顺带引用一句；\n"
        "- none：没有讲，或只是词语巧合。\n"
        "绝对不要因为出现了某个关键词就判 teach。必须以语义为准。\n"
        "只输出 JSON，不要输出 Markdown。"
    )
    user = (
        f"兴趣知识点：\n{interest_json}\n\n"
        f"最近转写（窗口 {window_start:.1f}s – {window_end:.1f}s）：\n{transcript_json}\n\n"
        "对每个兴趣输出一条判定。JSON 格式：\n"
        '{"judgments":[{"interest_id":"...","level":"teach|mention|none",'
        '"confidence":0.0,"start":0.0,"end":0.0,"quote":"原话",'
        '"reason":"一句话理由"}]}\n'
        "start/end 是这段讲解在本窗口转写中的起止秒数；没有讲就填 none，"
        "start/end 可以为 null。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return {}


def normalize_judgments(payload: dict, interests: list[Interest],
                        window_start: float, window_end: float) -> list[dict]:
    """Validate an LLM response and clamp window offsets to the request window."""
    known = {interest.id: interest for interest in interests}
    raw = payload.get("judgments") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        interest_id = str(item.get("interest_id") or item.get("id") or "")
        if interest_id not in known:
            continue
        level = str(item.get("level") or "none").strip().lower()
        if level in {"unrelated", "not_found", "not-teaching", "no"}:
            level = "none"
        if level not in LEVELS:
            level = "none"
        start = _as_float(item.get("start"), window_start)
        end = _as_float(item.get("end"), window_end)
        start = max(window_start, min(start, window_end))
        end = max(start + 0.1, min(end, window_end))
        result.append({
            "interest_id": interest_id,
            "level": level,
            "confidence": max(0.0, min(1.0, _as_float(item.get("confidence"), 0.0))),
            "start": round(start, 2),
            "end": round(end, 2),
            "quote": _clean(item.get("quote"), 600),
            "reason": _clean(item.get("reason"), 600),
        })
    # The contract asks for one judgement per interest. If the model omitted
    # some, treat the omission as "not teaching now" so an open topic can
    # close; otherwise a stale topic would stay open until max_span.
    seen = {item["interest_id"] for item in result}
    for interest in interests:
        if interest.id in seen:
            continue
        result.append({
            "interest_id": interest.id,
            "level": "none",
            "confidence": 0.0,
            "start": round(window_start, 2),
            "end": round(window_end, 2),
            "quote": "",
            "reason": "模型未返回该兴趣的判定，按未讲处理",
        })
    return result


class HostJudge:
    """File based judge channel: the host agent is the model."""

    def __init__(self, out_dir: Path, settings: SemanticSettings):
        self.out = Path(out_dir) / "judge"
        self.requests = self.out / "requests"
        self.responses = self.out / "responses"
        self.applied = self.out / "applied"
        self.pending_path = self.out / "pending.json"
        for path in (self.requests, self.responses, self.applied):
            path.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.seq = 0
        self.pending: dict[str, dict] = {}
        self.last_error = ""
        self._write_pending()

    def _write_pending(self) -> None:
        """One small file the host agent can read without globbing."""
        items = []
        for request_id in sorted(self.pending):
            payload = self.pending[request_id]
            items.append({
                "request_id": request_id,
                "path": str(self.requests / f"{request_id}.json"),
                "window": payload.get("window"),
                "interests": [
                    item.get("id") for item in payload.get("interests", [])
                ],
            })
        atomic_write_json(self.pending_path, {
            "pending": items,
            "updated_at": round(time.time(), 3),
        })

    def submit(self, interests: list[Interest], lines: list[dict],
               window_start: float, window_end: float) -> str:
        if len(self.pending) >= self.settings.max_pending:
            return ""
        self.seq += 1
        request_id = f"req_{self.seq:04d}"
        payload = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "created_at": round(time.time(), 3),
            "window": {"start": round(window_start, 2), "end": round(window_end, 2)},
            "interests": _interest_payload(interests, self.settings.max_interests),
            "transcript": _window_lines(lines, window_start, window_end),
            "instructions": (
                "用你自己的模型判断每个 interest 在这段课堂转写里是 teach / "
                "mention / none。teach 必须是正在定义、解释、推导、证明或完整例题；"
                "只出现关键词不算。把结果写到 judge/responses/<request_id>.json，"
                "格式：{\"request_id\":\"...\",\"judgments\":[{\"interest_id\":"
                "\"...\",\"level\":\"teach|mention|none\",\"confidence\":0.9,"
                "\"start\":0,\"end\":30,\"quote\":\"原话\",\"reason\":\"...\"}]}"
            ),
            "messages": build_prompt(
                interests, lines, window_start, window_end,
                self.settings.max_interests,
            ),
        }
        atomic_write_json(self.requests / f"{request_id}.json", payload)
        self.pending[request_id] = payload
        self._write_pending()
        return request_id

    def poll(self) -> list[dict]:
        events = []
        for path in sorted(self.responses.glob("req_*.json")):
            request_id = path.stem
            request = self.pending.get(request_id)
            if request is None:
                # Response for a request this process did not create (e.g. the
                # engine restarted). Load the request from disk if possible.
                request_path = self.requests / f"{request_id}.json"
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                self.last_error = f"judge response unreadable: {path.name}: {exc}"
                continue
            window = request.get("window") or {}
            interests = [
                Interest(
                    id=str(item.get("id")),
                    question=str(item.get("question", "")),
                    keywords=[str(word) for word in item.get("hints", [])],
                )
                for item in request.get("interests", [])
                if item.get("id")
            ]
            judgments = normalize_judgments(
                payload,
                interests,
                _as_float(window.get("start")),
                _as_float(window.get("end")),
            )
            events.append({
                "request_id": request_id,
                "window_start": _as_float(window.get("start")),
                "window_end": _as_float(window.get("end")),
                "judgments": judgments,
                "backend": "host",
            })
            try:
                path.replace(self.applied / path.name)
            except OSError:
                pass
            self.pending.pop(request_id, None)
        if events:
            self._write_pending()
        return events


class OpenAIJudge:
    """OpenAI compatible chat-completions backend."""

    def __init__(self, settings: SemanticSettings):
        self.settings = settings
        self.last_error = ""
        self._lock = threading.Lock()
        self._inflight = False
        self._results: list[dict] = []
        self.pending = 0
        self.seq = 0

    def available(self) -> bool:
        return bool(self.settings.openai_base_url and self.settings.openai_model)

    def submit(self, interests: list[Interest], lines: list[dict],
               window_start: float, window_end: float) -> str:
        if not self.available() or self._inflight:
            return ""
        self.seq += 1
        request_id = f"req_{self.seq:04d}"
        messages = build_prompt(
            interests, lines, window_start, window_end,
            self.settings.max_interests,
        )
        with self._lock:
            self._inflight = True
        thread = threading.Thread(
            target=self._call,
            args=(request_id, interests, window_start, window_end, messages),
            daemon=True,
        )
        thread.start()
        return request_id

    def _call(self, request_id: str, interests: list[Interest],
              window_start: float, window_end: float,
              messages: list[dict]) -> None:
        try:
            content = self._chat(messages)
            payload = _extract_json(content)
            if not payload:
                raise ValueError("model did not return JSON")
            judgments = normalize_judgments(
                payload, interests, window_start, window_end
            )
            event = {
                "request_id": request_id,
                "window_start": window_start,
                "window_end": window_end,
                "judgments": judgments,
                "backend": "openai",
            }
            with self._lock:
                self._results.append(event)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._inflight = False

    def _chat(self, messages: list[dict]) -> str:
        url = self.settings.openai_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1200,
        }
        if self.settings.openai_json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("empty choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")

    def poll(self) -> list[dict]:
        with self._lock:
            results = self._results
            self._results = []
        return results


class SemanticTrigger:
    """Facade used by the live engine."""

    def __init__(self, interests: list[Interest], out_dir: Path,
                 settings: SemanticSettings):
        self.interests = interests
        self.settings = settings
        self.last_request_at = -1e9
        self.last_window_end = 0.0
        self.last_text_len = 0
        self.requests = 0
        self.last_error = ""
        if settings.backend == "openai":
            self.backend = OpenAIJudge(settings)
            if not self.backend.available():
                self.last_error = (
                    "未配置 LLM 端点：设置 LECTURE_LLM_BASE_URL 和 "
                    "LECTURE_LLM_MODEL（可选 LECTURE_LLM_API_KEY）"
                )
        else:
            self.backend = HostJudge(out_dir, settings)

    def maybe_request(self, lines: list[dict], now_rel: float) -> str:
        if not self.settings.enabled or self.last_error:
            return ""
        if now_rel - self.last_request_at < self.settings.interval_seconds:
            return ""
        text_len = sum(len(str(line.get("text", ""))) for line in lines)
        if text_len <= self.last_text_len:
            return ""
        if text_len - self.last_text_len < self.settings.min_new_chars:
            return ""
        window_end = now_rel
        window_start = max(0.0, window_end - self.settings.window_seconds)
        if len(_window_lines(lines, window_start, window_end)) < 2:
            return ""
        request_id = self.backend.submit(
            self.interests, lines, window_start, window_end
        )
        if not request_id:
            return ""
        self.last_request_at = now_rel
        self.last_window_end = window_end
        self.last_text_len = text_len
        self.requests += 1
        return request_id

    def poll(self) -> list[dict]:
        events = self.backend.poll()
        error = getattr(self.backend, "last_error", "")
        if error:
            self.last_error = error
        return events

    def describe(self) -> dict:
        pending = len(getattr(self.backend, "pending", {}))
        if isinstance(self.backend, OpenAIJudge):
            pending = 1 if self.backend._inflight else 0
        return {
            "enabled": self.settings.enabled,
            "backend": self.settings.backend,
            "model": self.settings.openai_model if self.settings.backend == "openai" else "host-agent",
            "requests": self.requests,
            "pending": pending,
            "last_window_end": round(self.last_window_end, 2),
            "last_error": self.last_error,
        }