"""Unified dependency discovery and resolution for lecture-moment-finder.

One registry, one resolution order, one cache, one report:

    explicit CLI path / env var
      -> cached path (validated)
      -> PATH (shutil.which)
      -> bounded search in common user/app dirs
      -> fallback (e.g. PyAV for video)
      -> unavailable

All scripts should call this module instead of doing their own
`shutil.which` / `importlib.util.find_spec` checks.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE_VERSION = 1
CACHE_DIR = Path.home() / ".lecture-moment-finder"
CACHE_PATH = CACHE_DIR / "deps_cache.json"
MAX_RESULTS = 8

SKIP_DIRS = {
    "windows", "$recycle.bin", "system volume information", "node_modules",
    ".git", ".cache", "__pycache__", "windowsapps", "steamlibrary",
    "wegameapps", "temp", "tmp", "packages", "containers",
}

BINARY_SPECS: dict[str, dict] = {
    "ffmpeg": {
        "label": "ffmpeg",
        "names": ("ffmpeg.exe", "ffmpeg"),
        "env": ("LECTURE_FFMPEG", "FFMPEG_BINARY"),
        "required_for": "视频切片（可选；没有时用 PyAV）",
        "install": "https://ffmpeg.org/download.html 或 pip install av",
        "fallback": "PyAV",
        "search": True,
    },
}

MODULE_SPECS: dict[str, dict] = {
    "av": {
        "label": "PyAV",
        "module": "av",
        "pip": "av",
        "required_for": "实时录屏 / 视频切片兜底",
        "fallback": "",
    },
    "faster_whisper": {
        "label": "faster-whisper",
        "module": "faster_whisper",
        "pip": "faster-whisper",
        "required_for": "实时转写 / 本地字幕生成",
        "fallback": "外部 SRT/VTT",
    },
    "pyaudiowpatch": {
        "label": "pyaudiowpatch",
        "module": "pyaudiowpatch",
        "pip": "pyaudiowpatch",
        "required_for": "实时系统声音环回",
        "fallback": "麦克风（如果可用）",
    },
    "mss": {
        "label": "mss",
        "module": "mss",
        "pip": "mss",
        "required_for": "实时屏幕采集",
        "fallback": "不录屏，只录音",
    },
    "PIL": {
        "label": "Pillow",
        "module": "PIL",
        "pip": "pillow",
        "required_for": "屏幕帧 / 关键帧",
        "fallback": "",
    },
    "numpy": {
        "label": "numpy",
        "module": "numpy",
        "pip": "numpy",
        "required_for": "实时屏幕处理",
        "fallback": "",
    },
}

LIVE_MODULE_IDS = (
    "pyaudiowpatch", "mss", "PIL", "numpy", "faster_whisper", "av",
)

MODEL_SPECS: dict[str, dict] = {
    "semantic_model": {
        "label": "本地语义模型（可选）",
        "env": ("LECTURE_SEMANTIC_MODEL",),
        "default": "",
        "required_for": "本地语义 fallback（host 模式不需要）",
        "install": "可选；用 --model 或 LECTURE_SEMANTIC_MODEL 指定；host 模式不需要",
        "fallback": "host agent 模型",
    },
}


@dataclass
class Resolution:
    id: str
    label: str
    kind: str
    ok: bool
    path: str = ""
    version: str = ""
    source: str = ""
    detail: str = ""
    install: str = ""
    fallback: str = ""
    required_for: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _default_roots() -> list[Path]:
    home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    local_appdata = os.environ.get("LOCALAPPDATA")
    roots = [
        home,
        home / "Desktop",
        home / "Downloads",
        home / "Documents",
        Path(local_appdata) if local_appdata else home / "AppData" / "Local",
    ]
    for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    # dedupe while keeping order
    seen: set[str] = set()
    unique = []
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except OSError:
            key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _binary_version(path: str) -> str:
    try:
        result = subprocess.run(
            [path, "-version"], capture_output=True, text=True,
            timeout=8, check=False,
        )
        lines = (result.stdout or result.stderr or "").splitlines()
        return lines[0][:200] if lines else ""
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _module_version(module: str) -> str:
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        imported = __import__(module)
        return str(getattr(imported, "__version__", "") or "")
    except Exception:  # noqa: BLE001
        return ""


class Resolver:
    def __init__(
        self,
        *,
        search: bool = True,
        max_seconds: float = 6.0,
        max_depth: int = 5,
        cache_path: str | Path = CACHE_PATH,
        roots: list[str | Path] | None = None,
    ):
        self.search_enabled = bool(search)
        self.max_seconds = max(0.5, float(max_seconds))
        self.max_depth = max(1, int(max_depth))
        self.cache_path = Path(cache_path)
        self.roots = [Path(r) for r in roots] if roots else _default_roots()
        self.cache = self._load_cache()
        self._resolved: dict[str, Resolution] = {}

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> dict:
        if not self.cache_path.is_file():
            return {"version": CACHE_VERSION, "binaries": {}}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": CACHE_VERSION, "binaries": {}}
        if data.get("version") != CACHE_VERSION:
            return {"version": CACHE_VERSION, "binaries": {}}
        return data

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError:
            pass

    def _cached_binary(self, dep_id: str) -> str:
        entry = (self.cache.get("binaries") or {}).get(dep_id) or {}
        path = Path(str(entry.get("path") or ""))
        if not path.is_file():
            return ""
        try:
            stat = path.stat()
        except OSError:
            return ""
        if entry.get("size") and entry["size"] != stat.st_size:
            return ""
        if entry.get("mtime") and abs(entry["mtime"] - stat.st_mtime) > 1:
            return ""
        return str(path)

    def _remember_binary(self, dep_id: str, path: str, source: str) -> None:
        try:
            stat = Path(path).stat()
            self.cache.setdefault("binaries", {})[dep_id] = {
                "path": path,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "source": source,
                "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_cache()
        except OSError:
            pass

    # --------------------------------------------------------------- binaries
    def _search_binary(self, dep_id: str) -> list[str]:
        spec = BINARY_SPECS[dep_id]
        names = {name.lower() for name in spec["names"]}
        results: list[tuple[int, str]] = []
        seen: set[str] = set()
        started = time.monotonic()

        for root in self.roots:
            if not root.is_dir():
                continue
            if time.monotonic() - started > self.max_seconds:
                break
            root_depth = len(root.parts)
            for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
                if time.monotonic() - started > self.max_seconds:
                    break
                current_path = Path(current)
                depth = len(current_path.parts) - root_depth
                if depth >= self.max_depth:
                    dirs[:] = []
                else:
                    dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
                for name in files:
                    if name.lower() not in names:
                        continue
                    path = current_path / name
                    key = str(path).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0
                    results.append((size, str(path)))
                    if len(results) >= MAX_RESULTS:
                        break
                if len(results) >= MAX_RESULTS:
                    break
            if len(results) >= MAX_RESULTS:
                break

        # prefer larger binaries: tiny bundled ffmpeg stubs often lack encoders
        results.sort(key=lambda item: -item[0])
        return [path for _size, path in results]

    def binary(self, dep_id: str, explicit: str | Path | None = None,
               search: bool | None = None) -> Resolution:
        spec = BINARY_SPECS.get(dep_id)
        if not spec:
            return Resolution(dep_id, dep_id, "binary", False,
                              detail=f"unknown binary dependency: {dep_id}")
        if dep_id in self._resolved:
            return self._resolved[dep_id]

        candidates: list[tuple[str, str]] = []
        if explicit:
            candidates.append((str(explicit), "explicit"))
        for env_name in spec.get("env", ()):
            value = os.environ.get(env_name)
            if value:
                candidates.append((value, f"env:{env_name}"))
        cached = self._cached_binary(dep_id)
        if cached:
            candidates.append((cached, "cache"))
        for name in spec["names"]:
            found = shutil.which(name)
            if found:
                candidates.append((found, "PATH"))

        for value, source in candidates:
            path = Path(str(value)).expanduser()
            if path.is_file():
                resolution = Resolution(
                    id=dep_id,
                    label=spec["label"],
                    kind="binary",
                    ok=True,
                    path=str(path),
                    version=_binary_version(str(path)),
                    source=source,
                    install=spec.get("install", ""),
                    fallback=spec.get("fallback", ""),
                    required_for=spec.get("required_for", ""),
                )
                self._remember_binary(dep_id, str(path), source)
                self._resolved[dep_id] = resolution
                return resolution

        if search is None:
            search = self.search_enabled
        if search and spec.get("search"):
            found_paths = self._search_binary(dep_id)
            if found_paths:
                path = found_paths[0]
                resolution = Resolution(
                    id=dep_id,
                    label=spec["label"],
                    kind="binary",
                    ok=True,
                    path=path,
                    version=_binary_version(path),
                    source="search",
                    install=spec.get("install", ""),
                    fallback=spec.get("fallback", ""),
                    required_for=spec.get("required_for", ""),
                )
                self._remember_binary(dep_id, path, "search")
                self._resolved[dep_id] = resolution
                return resolution

        resolution = Resolution(
            id=dep_id,
            label=spec["label"],
            kind="binary",
            ok=False,
            detail="not found in explicit/env/cache/PATH/common dirs",
            install=spec.get("install", ""),
            fallback=spec.get("fallback", ""),
            required_for=spec.get("required_for", ""),
        )
        self._resolved[dep_id] = resolution
        return resolution

    # ---------------------------------------------------------------- modules
    def module(self, dep_id: str) -> Resolution:
        spec = MODULE_SPECS.get(dep_id)
        if not spec:
            return Resolution(dep_id, dep_id, "module", False,
                              detail=f"unknown module dependency: {dep_id}")
        if dep_id in self._resolved:
            return self._resolved[dep_id]
        module = spec["module"]
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        resolution = Resolution(
            id=dep_id,
            label=spec["label"],
            kind="module",
            ok=found,
            version=_module_version(module) if found else "",
            source="import" if found else "",
            detail="" if found else f"python module not installed: {module}",
            install=f"pip install {spec['pip']}" if spec.get("pip") else "",
            fallback=spec.get("fallback", ""),
            required_for=spec.get("required_for", ""),
        )
        self._resolved[dep_id] = resolution
        return resolution

    # ----------------------------------------------------------------- models
    def model(self, dep_id: str, explicit: str | Path | None = None) -> Resolution:
        spec = MODEL_SPECS.get(dep_id)
        if not spec:
            return Resolution(dep_id, dep_id, "model", False,
                              detail=f"unknown model dependency: {dep_id}")
        candidates: list[tuple[str, str]] = []
        if explicit:
            candidates.append((str(explicit), "explicit"))
        for env_name in spec.get("env", ()):
            value = os.environ.get(env_name)
            if value:
                candidates.append((value, f"env:{env_name}"))
        if spec.get("default"):
            candidates.append((str(spec["default"]), "default"))
        for value, source in candidates:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return Resolution(
                    id=dep_id, label=spec["label"], kind="model", ok=True,
                    path=str(path), source=source,
                    required_for=spec.get("required_for", ""),
                    fallback=spec.get("fallback", ""),
                )
        return Resolution(
            id=dep_id, label=spec["label"], kind="model", ok=False,
            detail="model file not found",
            install=spec.get("install", ""),
            fallback=spec.get("fallback", ""),
            required_for=spec.get("required_for", ""),
        )

    # ----------------------------------------------------------------- report
    def video_engine(self, ffmpeg_explicit: str | Path | None = None) -> dict:
        ffmpeg = self.binary("ffmpeg", explicit=ffmpeg_explicit)
        pyav = self.module("av")
        if ffmpeg.ok:
            engine = "ffmpeg"
        elif pyav.ok:
            engine = "pyav"
        else:
            engine = "none"
        return {"engine": engine, "ffmpeg": ffmpeg.to_dict(),
                "pyav": pyav.to_dict()}

    def report(self) -> dict:
        binaries = [self.binary(dep_id).to_dict() for dep_id in BINARY_SPECS]
        modules = [self.module(dep_id).to_dict() for dep_id in MODULE_SPECS]
        models = [self.model(dep_id).to_dict() for dep_id in MODEL_SPECS]
        return {
            "binaries": binaries,
            "modules": modules,
            "models": models,
            "video_engine": self.video_engine()["engine"],
            "live_ready": all(self.module(dep_id).ok for dep_id in LIVE_MODULE_IDS),
        }


_GLOBAL: Resolver | None = None


def get_resolver(**kwargs) -> Resolver:
    """Return a process-wide resolver; later kwargs update search policy."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Resolver(**kwargs)
    else:
        if "search" in kwargs:
            _GLOBAL.search_enabled = bool(kwargs["search"])
        if "max_seconds" in kwargs:
            _GLOBAL.max_seconds = max(0.5, float(kwargs["max_seconds"]))
        if "roots" in kwargs and kwargs["roots"]:
            _GLOBAL.roots = [Path(r) for r in kwargs["roots"]]
    return _GLOBAL


def search_binaries(dep_id: str, *, roots=None, max_depth: int = 5,
                    max_seconds: float = 20.0) -> list[str]:
    """Public helper for compatibility scripts (e.g. discover_ffmpeg)."""
    resolver = Resolver(search=False, roots=roots, max_depth=max_depth,
                        max_seconds=max_seconds)
    if dep_id not in BINARY_SPECS:
        return []
    return resolver._search_binary(dep_id)
