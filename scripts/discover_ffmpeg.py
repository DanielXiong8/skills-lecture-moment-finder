#!/usr/bin/env python3
"""Compatibility wrapper: find ffmpeg via the unified dependency resolver.

Prefer `python scripts/doctor.py --json` for a full dependency report.
This script is kept for users/agents that only need ffmpeg candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deps import get_resolver, search_binaries  # noqa: E402


def _version(path: str) -> str:
    from deps import _binary_version
    return _binary_version(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="查找 ffmpeg（兼容入口；推荐用 doctor.py）。")
    parser.add_argument("--root", action="append", default=[],
                        help="额外搜索根目录，可重复；不传则用常见目录")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="最大目录深度（默认 5）")
    parser.add_argument("--max-seconds", type=float, default=20.0,
                        help="搜索时间上限（默认 20 秒）")
    parser.add_argument("--verify", action="store_true",
                        help="对找到的 ffmpeg 跑 -version 验证")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    resolver = get_resolver(search=False)
    direct = resolver.binary("ffmpeg")
    paths: list[str] = []
    if direct.ok and direct.path:
        paths.append(direct.path)
    for path in search_binaries(
            "ffmpeg", roots=args.root or None,
            max_depth=max(1, args.max_depth),
            max_seconds=max(0.5, args.max_seconds)):
        if path not in paths:
            paths.append(path)

    results = []
    for path in paths[:8]:
        try:
            size_mb = round(Path(path).stat().st_size / 1024 / 1024, 1)
        except OSError:
            size_mb = 0.0
        item = {"path": path, "size_mb": size_mb}
        if args.verify:
            item["version"] = _version(path)
        results.append(item)
    results.sort(key=lambda item: -item["size_mb"])

    if args.json:
        print(json.dumps({"found": results, "count": len(results)},
                         ensure_ascii=False, indent=2))
        return 0 if results else 1

    if not results:
        print("未找到 ffmpeg。可以指定 --root 再搜，或直接安装 PyAV（av）：")
        print("  python scripts/ensure_deps.py")
        return 1
    for item in results:
        line = f"[OK] {item['path']}  {item['size_mb']} MB"
        if item.get("version"):
            line += f"  {item['version']}"
        print(line)
    print("\n传给 Skill：")
    print(f'  python scripts/find_moment.py ... --ffmpeg "{results[0]["path"]}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())