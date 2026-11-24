#!/usr/bin/env python3
"""Host-side helper for live semantic judging.

The live engine writes ``judge/requests/req_0001.json``. The host agent (the
model that runs this Skill) reads the request, judges with its own model, and
writes the response back. This script only moves files and validates the shape;
it never calls a model and never uses keyword matching.

Typical loop inside CodeArts::

    python scripts/judge_live.py next --out <session>
    # agent reads the request and writes /tmp/resp_0001.json
    python scripts/judge_live.py submit --out <session> \
        --request req_0001 --file /tmp/resp_0001.json

Run ``next`` again; it blocks until the engine produces the next window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_core import atomic_write_json  # noqa: E402

PROTOCOL = "lecture-live-judge/1"


def _dirs(out: Path) -> tuple[Path, Path, Path]:
    root = out / "judge"
    return root / "requests", root / "responses", root / "applied"


def _pending(out: Path) -> list[Path]:
    requests, responses, applied = _dirs(out)
    if not requests.is_dir():
        return []
    done = {path.name for path in responses.glob("req_*.json")}
    done |= {path.name for path in applied.glob("req_*.json")}
    return [
        path for path in sorted(requests.glob("req_*.json"))
        if path.name not in done
    ]


def cmd_status(args) -> int:
    out = Path(args.out)
    requests, responses, applied = _dirs(out)
    pending = _pending(out)
    print(json.dumps({
        "out": str(out),
        "requests": len(list(requests.glob("req_*.json"))) if requests.is_dir() else 0,
        "responses": len(list(responses.glob("req_*.json"))) if responses.is_dir() else 0,
        "applied": len(list(applied.glob("req_*.json"))) if applied.is_dir() else 0,
        "pending": [path.stem for path in pending],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_next(args) -> int:
    out = Path(args.out)
    deadline = time.time() + max(0.0, args.timeout)
    while True:
        pending = _pending(out)
        if pending:
            path = pending[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                print(f"无法读取 {path}: {exc}", file=sys.stderr)
                return 2
            print(json.dumps({
                "request_path": str(path),
                "request": payload,
            }, ensure_ascii=False, indent=2))
            return 0
        if time.time() >= deadline:
            print(json.dumps({"pending": [], "timeout": True}, ensure_ascii=False))
            return 0
        time.sleep(max(0.2, args.interval))


def cmd_submit(args) -> int:
    out = Path(args.out)
    requests, responses, _applied = _dirs(out)
    request_id = args.request.strip()
    if request_id.endswith(".json"):
        request_id = request_id[:-5]
    request_path = requests / f"{request_id}.json"
    if not request_path.is_file():
        print(f"找不到请求：{request_path}", file=sys.stderr)
        return 2
    response_path = Path(args.file)
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"无法读取判定文件 {response_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("判定文件必须是 JSON 对象", file=sys.stderr)
        return 2
    judgments = payload.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        print("判定文件缺少非空 judgments 列表", file=sys.stderr)
        return 2
    for index, item in enumerate(judgments, 1):
        if not isinstance(item, dict):
            print(f"judgments[{index}] 不是对象", file=sys.stderr)
            return 2
        if not item.get("interest_id") or item.get("level") not in {
            "teach", "mention", "none"
        }:
            print(
                f"judgments[{index}] 需要 interest_id 和 "
                "level=teach|mention|none",
                file=sys.stderr,
            )
            return 2
    payload["protocol"] = PROTOCOL
    payload["request_id"] = request_id
    payload["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    responses.mkdir(parents=True, exist_ok=True)
    target = responses / f"{request_id}.json"
    atomic_write_json(target, payload)
    print(f"已提交：{target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实时语义判定的宿主侧助手。")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="查看待判定/已判定数量")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("next", help="等待并打印下一个判定请求")
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="最多等待秒数；0=立即返回")
    p.add_argument("--interval", type=float, default=2.0)
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("submit", help="提交宿主模型的判定结果")
    p.add_argument("--out", required=True)
    p.add_argument("--request", required=True, help="req_0001 或 req_0001.json")
    p.add_argument("--file", required=True, help="判定结果 JSON 文件")
    p.set_defaults(func=cmd_submit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())