#!/usr/bin/env python3
"""One dependency report for the whole Skill.

    python scripts/doctor.py
    python scripts/doctor.py --json
    python scripts/doctor.py --no-search
    python scripts/doctor.py --search-seconds 20

The resolver is shared by find_moment / transcribe / live_ctl / ensure_deps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deps import BINARY_SPECS, LIVE_MODULE_IDS, MODULE_SPECS, get_resolver  # noqa: E402


def _status(item: dict) -> str:
    if item.get("ok"):
        return "OK"
    return "MISSING"


def _print_table(report: dict) -> int:
    print(f"python: {sys.executable} ({sys.version.split()[0]})")
    print("\n== 二进制 ==")
    for item in report["binaries"]:
        line = f"{item['label']:<10} {_status(item):<8} {item.get('path') or item.get('detail', '')}"
        if item.get("version"):
            line += f"  {item['version']}"
        if not item.get("ok"):
            line += f"  -> fallback: {item.get('fallback') or '无'}"
            if item.get("install"):
                line += f"；安装：{item['install']}"
        print(line)

    print("\n== Python 模块 ==")
    for item in report["modules"]:
        line = f"{item['label']:<16} {_status(item):<8} {item.get('version') or item.get('detail', '')}"
        if not item.get("ok"):
            if item.get("install"):
                line += f"  -> {item['install']}"
            if item.get("fallback"):
                line += f"；fallback: {item['fallback']}"
        print(line)

    print("\n== 本地模型 ==")
    for item in report["models"]:
        line = f"{item['label']:<28} {_status(item):<8} {item.get('path') or item.get('detail', '')}"
        if not item.get("ok") and item.get("fallback"):
            line += f"  -> fallback: {item['fallback']}"
        print(line)

    print("\n== 汇总 ==")
    engine = report["video_engine"]
    print(f"视频切片引擎: {engine} "
          f"({'ffmpeg' if engine == 'ffmpeg' else 'PyAV fallback' if engine == 'pyav' else '不可用'})")
    print(f"实时模式依赖: {'齐全' if report['live_ready'] else '不齐全'}")

    missing = [item for item in report["modules"] if not item.get("ok")]
    if missing:
        print("\n缺少实时依赖，运行一次引导：")
        print(f"  {sys.executable} scripts/ensure_deps.py --model base [--mirror]")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 Skill 的全部依赖。")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--no-search", action="store_true",
                        help="不搜索常见目录，只查显式/env/缓存/PATH")
    parser.add_argument("--search-seconds", type=float, default=20.0,
                        help="常见目录搜索时间上限（默认 20 秒）")
    parser.add_argument("--root", action="append", default=[],
                        help="额外搜索根目录，可重复")
    args = parser.parse_args(argv)

    resolver = get_resolver(
        search=not args.no_search,
        max_seconds=args.search_seconds,
        roots=args.root or None,
    )
    report = resolver.report()

    if args.json:
        print(json.dumps({
            "python": sys.executable,
            "version": sys.version.split()[0],
            "report": report,
            "live_module_ids": list(LIVE_MODULE_IDS),
            "binary_ids": list(BINARY_SPECS),
            "module_ids": list(MODULE_SPECS),
        }, ensure_ascii=False, indent=2))
        return 0 if report["live_ready"] else 1

    return _print_table(report)


if __name__ == "__main__":
    raise SystemExit(main())