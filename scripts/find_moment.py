#!/usr/bin/env python3
"""CLI: locate the full stretch a lecture covers a topic and write replay outputs.

Adds an optional semantic hit rule:
  * keyword/must-include recall of candidate windows
  * local Qwen judge: "is this teaching the knowledge point, or only mentioning it?"
  * no-answer when every candidate is rejected
  * teaching window from the judge, not a fixed keyword-score window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moment_core import TOOL_NAME, build_outputs, parse_transcript, search  # noqa: E402
from semantic_core import (  # noqa: E402
    SemanticJudge,
    apply_host_response,
    apply_host_response_corpus,
    load_specs,
    load_transcript_dir,
    prepare_host_request_legacy,
    prepare_host_request_corpus,
    semantic_search,
    spec_from_query,
)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}小时{minutes}分{sec}秒"
    return f"{minutes}分{sec}秒" if minutes else f"{sec}秒"


def _load_spec(args) -> dict:
    if args.spec:
        data = json.loads(Path(args.spec).read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "specs" in data:
            data = data["specs"]
        if isinstance(data, list):
            if not args.knowledge_point:
                raise SystemExit("--spec 是列表时必须同时给 --knowledge-point")
            for item in data:
                if str(item.get("id")) == args.knowledge_point:
                    return item
            raise SystemExit(f"spec 列表里没有 {args.knowledge_point}")
        return data
    if args.specs:
        specs = load_specs(args.specs)
        if not args.knowledge_point:
            raise SystemExit("--specs 必须同时给 --knowledge-point")
        if args.knowledge_point not in specs:
            raise SystemExit(f"{args.specs} 里没有 {args.knowledge_point}")
        return specs[args.knowledge_point]
    keywords = [item.strip() for item in (args.keywords or "").split(",") if item.strip()]
    return spec_from_query(args.query, keywords)


def _write_ai_fallback(out_dir: Path, query: str, spec: dict,
                       reason: str, response_path: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "protocol": "lecture-ai-fallback/1",
        "query": query,
        "reason": reason,
        "knowledge_point": {
            "id": spec.get("id", ""),
            "name": spec.get("name", query),
            "definition": spec.get("definition", ""),
            "must_include": spec.get("must_include", []),
            "mention_only": spec.get("mention_only", ""),
        },
        "instructions": (
            "课程录播中没有找到这个知识点的完整教学段。请用你自己的知识给出简明讲解，"
            "并明确标注为 AI 补充；不要编造课堂原话、时间戳或老师观点。"
        ),
        "response_format": {
            "title": "知识点名称",
            "explanation": "简明讲解，说明定义/条件/结论或方法",
            "key_points": ["关键步骤 1", "关键步骤 2"],
            "example": "可选：一个最小例子",
            "caveat": "课程录播中未找到该知识点的完整讲解，以上为 AI 补充。",
        },
    }
    request_path = out_dir / "ai_fallback_request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fallback = {
        "mode": "ai",
        "source": "ai",
        "warning": "课程录播中未找到该知识点的完整讲解；以下为 AI 补充，不是课堂原话，也没有时间戳。",
        "request": str(request_path),
    }
    if response_path:
        response = json.loads(
            Path(response_path).read_text(encoding="utf-8-sig")
        )
        if not isinstance(response, dict):
            raise ValueError("AI fallback response must be a JSON object")
        fallback["response"] = response
        result_path = out_dir / "result.json"
        result_path.write_text(json.dumps({
            "query": query,
            "decision": "not_found",
            "reason": reason,
            "fallback": fallback,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        fallback["result"] = str(result_path)
    return fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从课堂录音/录屏的转写文本中定位老师讲某个知识点的完整段落。"
    )
    parser.add_argument("--transcript", default="",
                        help="转写文件：.json/.srt/.vtt/.txt/.md")
    parser.add_argument("--transcripts-dir", default="",
                        help="课程转写目录；跨讲检索，每个文件一讲")
    parser.add_argument("--query", required=True,
                        help="自然语言问题，例如：老师讲动态规划状态转移方程的那几分钟")
    parser.add_argument("--out", default="out", help="输出目录（默认 ./out）")
    parser.add_argument("--media", default="", help="可选：原始录音/录屏，用于截取片段")
    parser.add_argument("--ffmpeg", default="",
                        help="可选：ffmpeg.exe 路径；PATH 里没有时用它（也可用 LECTURE_FFMPEG）")
    parser.add_argument("--no-deps-search", action="store_true",
                        help="不搜索常见目录；只查显式/env/缓存/PATH 和 PyAV fallback")
    parser.add_argument("--top", type=int, default=3, help="返回片段数（默认 3）")
    parser.add_argument("--min-span", type=float, default=45.0,
                        help="完整讨论的最短秒数（默认 45）")
    parser.add_argument("--max-span", type=float, default=0.0,
                        help="完整讨论的最长秒数；0=按录音长度自动（默认 0）")
    parser.add_argument("--core-min", type=float, default=30.0,
                        help="核心片段最短秒数（默认 30）")
    parser.add_argument("--core-max", type=float, default=90.0,
                        help="核心片段最长秒数（默认 90）")
    parser.add_argument("--rel-threshold", type=float, default=0.15,
                        help="继续扩展的相关度阈值，相对峰值（默认 0.15）")
    parser.add_argument("--bridge", type=int, default=2,
                        help="允许跨过几个低相关句（默认 2）")
    parser.add_argument("--max-gap", type=float, default=20.0,
                        help="相邻句超过多少秒视为换话题（默认 20）")
    parser.add_argument("--no-save", "--no-pack", dest="no_save", action="store_true",
                        help="只输出结果，不写 result/report/clip 文件")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印结果")
    # semantic rule
    parser.add_argument("--semantic", choices=["auto", "off", "llm", "host"], default="auto",
                        help="语义命中规则：auto=有 host 请求/响应文件就用 host，否则关键词；"
                             "off=关键词；host=由宿主 agent 的模型判定；llm=本地 Qwen fallback")
    parser.add_argument("--spec", default="", help="单个知识点契约 JSON")
    parser.add_argument("--specs", default="", help="多个知识点契约 JSON 文件")
    parser.add_argument("--knowledge-point", default="", help="从 --specs 里选哪个 id")
    parser.add_argument("--keywords", default="", help="逗号分隔的关键词（无 spec 时用）")
    parser.add_argument("--model", default="", help="本地语义模型路径（GGUF）")
    parser.add_argument("--judge-request-out", default="",
                        help="host 模式：把判定请求写到这个 JSON")
    parser.add_argument("--judge-response", default="",
                        help="host 模式：宿主模型判定后的 JSON")
    parser.add_argument("--max-candidates", type=int, default=300,
                        help="语义判定最多看几个候选（默认 300）")
    parser.add_argument("--per-lecture-candidates", type=int, default=12,
                        help="跨讲模式每讲最多取几个候选（默认 12）")
    parser.add_argument("--bridge-terms", default="",
                        help="LLM 推断出的等价说法/前置知识点，逗号分隔；用于二次检索")
    parser.add_argument("--retry-terms", default="",
                        help="第二轮：宿主首轮给出的同义/错写表述，逗号分隔；重建候选池再判定")
    parser.add_argument("--no-retry", action="store_true",
                        help="关闭宿主 retry 轮（首轮无 teach 时不再请求第二轮）")
    parser.add_argument("--ai-fallback", action="store_true",
                        help="课程里没找到教学段时，生成 AI 补充讲解请求（不冒充课堂证据）")
    parser.add_argument("--ai-fallback-response", default="",
                        help="宿主/LLM 写好的 AI 补充讲解 JSON")
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="候选必须覆盖多少必含项才送语义判定（默认 0.5）")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="语义判定置信度阈值（默认 0.6）")
    parser.add_argument("--min-verify-confidence", type=float, default=0.6,
                        help="二次命题确认的置信度阈值（默认 0.6）")
    args = parser.parse_args(argv)

    bridge_terms = [
        item.strip() for item in (args.bridge_terms or "").split(",") if item.strip()
    ]
    retry_terms = [
        item.strip() for item in (args.retry_terms or "").split(",") if item.strip()
    ]
    if retry_terms:
        bridge_terms = bridge_terms + retry_terms
    docs = []
    turns = []
    if args.transcripts_dir:
        try:
            docs = load_transcript_dir(args.transcripts_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[{TOOL_NAME}] 无法读取转写目录：{exc}", file=sys.stderr)
            return 2
        turns = docs[0].turns
    else:
        if not args.transcript:
            print(f"[{TOOL_NAME}] 需要 --transcript 或 --transcripts-dir",
                  file=sys.stderr)
            return 2
        try:
            _meta, turns = parse_transcript(args.transcript)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[{TOOL_NAME}] 无法读取转写：{exc}", file=sys.stderr)
            return 2

    semantic_result = None
    segments: list[dict] = []
    notes: list[str] = []

    if args.semantic == "host" or (
            args.semantic == "auto" and (args.judge_response or args.judge_request_out)):
        try:
            spec = _load_spec(args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[{TOOL_NAME}] 无法读取知识点契约：{exc}", file=sys.stderr)
            return 2
        if bridge_terms:
            spec["bridge_terms"] = bridge_terms
        if retry_terms:
            spec["must_include"] = list(spec.get("must_include", [])) + retry_terms
            spec["retry_terms"] = retry_terms
        if docs:
            request = prepare_host_request_corpus(
                docs, spec,
                max_candidates=max(1, args.max_candidates),
                per_doc=max(1, args.per_lecture_candidates),
                min_coverage=max(0.0, args.min_coverage),
                bridge_terms=bridge_terms,
            )
        else:
            request = prepare_host_request_legacy(
                turns, spec,
                max_candidates=min(6, max(1, args.max_candidates)),
                bridge_terms=bridge_terms,
            )
        if not args.judge_response:
            request_path = (Path(args.judge_request_out) if args.judge_request_out
                            else Path(args.out) / (
                                "judge_request_retry.json" if retry_terms
                                else "judge_request.json"))
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{TOOL_NAME}] 已生成宿主判定请求：{request_path}"
                  f"（{len(request.get('candidates', []))} 个候选）")
            print("请用宿主 agent 的模型按 instructions 判定，把 JSON 通过 "
                  "--judge-response 传回。")
            return 0
        try:
            response = json.loads(Path(args.judge_response).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[{TOOL_NAME}] 无法读取宿主判定：{exc}", file=sys.stderr)
            return 2
        if docs:
            semantic_result = apply_host_response_corpus(
                docs, spec, request, response,
                core_min=args.core_min, core_max=args.core_max,
            )
        else:
            semantic_result = apply_host_response(
                turns, spec, request, response,
                core_min=args.core_min, core_max=args.core_max,
            )
        segments = semantic_result["segments"]
        notes.append(f"宿主语义判定：{semantic_result['decision']}；"
                     f"{semantic_result['reason']}")
        if (not docs and semantic_result["decision"] != "teach"
                and not retry_terms and not args.no_retry):
            suggested = response.get("retry_terms") or []
            hint_path = Path(args.out) / "judge_retry_hint.json"
            hint_path.parent.mkdir(parents=True, exist_ok=True)
            hint_path.write_text(json.dumps({
                "round": 1,
                "decision": semantic_result["decision"],
                "retry_terms_suggested": suggested,
                "next": "用 --retry-terms \"逗号分隔\" 重跑并传第二轮 --judge-response；"
                        "第二轮仍无 teach 才进入 AI 补讲",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{TOOL_NAME}] 首轮无 teach；宿主 retry_terms={suggested}；"
                  f"第二轮入口见 {hint_path}（不直接降级）。")
            return 0
    elif args.semantic == "llm":
        if docs:
            print(f"[{TOOL_NAME}] llm 模式暂不支持 --transcripts-dir；"
                  "跨讲请用 --semantic host", file=sys.stderr)
            return 2
        try:
            spec = _load_spec(args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[{TOOL_NAME}] 无法读取知识点契约：{exc}", file=sys.stderr)
            return 2
        judge = SemanticJudge(model_path=args.model or None)
        if judge.available:
            semantic_result = semantic_search(
                turns,
                spec,
                judge,
                max_candidates=max(1, args.max_candidates),
                min_coverage=args.min_coverage,
                min_confidence=args.min_confidence,
                min_verify_confidence=args.min_verify_confidence,
                core_min=args.core_min,
                core_max=args.core_max,
            )
            segments = semantic_result["segments"]
            notes.append(f"语义判定：{semantic_result['decision']}；"
                         f"{semantic_result['reason']}")
        else:
            notes.append(f"语义后端不可用（{judge.error}），回退关键词匹配。")
            if args.semantic == "llm":
                print(f"[{TOOL_NAME}] 语义后端不可用：{judge.error}", file=sys.stderr)
                return 2

    if semantic_result is None:
        query = args.query + (" " + " ".join(bridge_terms) if bridge_terms else "")
        if docs:
            merged = []
            for doc in docs:
                for segment in search(
                    doc.turns,
                    query,
                    top=1,
                    min_span=args.min_span,
                    max_span=args.max_span,
                    core_min=args.core_min,
                    core_max=args.core_max,
                    rel_threshold=args.rel_threshold,
                    bridge=args.bridge,
                    max_gap=args.max_gap,
                ):
                    segment = dict(segment)
                    segment["source"] = {
                        "slug": doc.slug, "date": doc.date, "part": doc.part,
                        "path": str(doc.path),
                    }
                    merged.append(segment)
            merged.sort(
                key=lambda item: float(
                    item.get("score") or item.get("raw_score") or 0.0
                ),
                reverse=True,
            )
            segments = merged[:max(1, args.top)]
        else:
            segments = search(
                turns,
                query,
                top=max(1, args.top),
                min_span=args.min_span,
                max_span=args.max_span,
                core_min=args.core_min,
                core_max=args.core_max,
                rel_threshold=args.rel_threshold,
                bridge=args.bridge,
                max_gap=args.max_gap,
            )

    if not segments:
        payload = {
            "query": args.query,
            "corpus": bool(docs),
            "decision": (semantic_result or {}).get("decision", "not_found"),
            "reason": (semantic_result or {}).get("reason", "没有找到匹配片段。"),
            "candidates": (semantic_result or {}).get("candidates", []),
            "selection_reason": (semantic_result or {}).get("selection_reason", ""),
            "bridge_terms": bridge_terms,
        }
        fallback = None
        if args.ai_fallback or args.ai_fallback_response:
            try:
                fallback_spec = _load_spec(args)
            except (OSError, ValueError, json.JSONDecodeError):
                fallback_spec = spec_from_query(args.query, bridge_terms)
            fallback = _write_ai_fallback(
                Path(args.out), args.query, fallback_spec,
                payload["reason"], args.ai_fallback_response,
            )
            payload["ai_fallback"] = fallback
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[{TOOL_NAME}] 未找到教学段：{payload['reason']}")
            for item in payload["candidates"]:
                print(f"  - [{item.get('source', '')}] "
                      f"{item['start_text']}-{item['end_text']} "
                      f"teach={item['teach']} conf={item['confidence']} "
                      f"{item['reason'][:80]}")
            if fallback:
                print(f"[{TOOL_NAME}] AI 补充讲解请求：{fallback['request']}")
                if fallback.get("response"):
                    response = fallback["response"]
                    print(f"  {response.get('title', '')}")
                    print(f"  {response.get('explanation', '')[:500]}")
                print(f"  {fallback['warning']}")
        return 0 if fallback else 1

    ai_fallback = None
    if ((args.ai_fallback or args.ai_fallback_response)
            and semantic_result is not None
            and semantic_result.get("decision") != "teach"):
        try:
            fallback_spec = _load_spec(args)
        except (OSError, ValueError, json.JSONDecodeError):
            fallback_spec = spec_from_query(args.query, bridge_terms)
        ai_fallback = _write_ai_fallback(
            Path(args.out), args.query, fallback_spec,
            "只找到提及/未完整讲解，没有完整教学段；AI 补充讲解如下。",
            args.ai_fallback_response,
        )

    outputs = None
    if not args.no_save:
        if docs:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "query": args.query,
                "corpus": True,
                "bridge_terms": bridge_terms,
                "segments": segments,
                "semantic": semantic_result,
                "ai_fallback": ai_fallback,
                "sources": [
                    {"slug": doc.slug, "date": doc.date, "part": doc.part}
                    for doc in docs
                ],
                "notes": notes,
            }
            result_path = out_dir / "result.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            lines = [f"# {args.query}", ""]
            for index, segment in enumerate(segments, 1):
                source = segment.get("source") or {}
                lines.append(
                    f"{index}. [{source.get('slug', '')}] "
                    f"{segment['start_text']} – {segment['end_text']}；"
                    f"核心 {segment['core_start_text']} – {segment['core_end_text']}"
                )
                lines.append(f"   {segment.get('quote', '')[:160]}")
            report_path = out_dir / "report.md"
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            outputs = {"out_dir": str(out_dir), "report": str(report_path)}
            notes.append("跨讲模式：result.json 的每个 segment 带 source，"
                         "媒体切片请用该讲的 --transcript/--media 单独生成。")
        else:
            if not args.media:
                notes.append("未提供 --media，未生成可播放片段；时间戳与转写仍然可复核。")
            outputs = build_outputs(
                args.out,
                query=args.query,
                transcript_path=args.transcript,
                segments=segments,
                media_path=args.media or None,
                notes=notes,
                ffmpeg_path=args.ffmpeg or None,
                search_deps=not args.no_deps_search,
            )

    if semantic_result is not None and not args.no_save and not docs:
        result_path = Path(args.out) / "result.json"
        try:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        saved["semantic"] = {
            "decision": semantic_result["decision"],
            "reason": semantic_result["reason"],
            "candidates": semantic_result["candidates"],
            "selection_reason": semantic_result.get("selection_reason", ""),
        }
        if ai_fallback:
            saved["ai_fallback"] = ai_fallback
        result_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    if args.json:
        print(json.dumps({
            "segments": segments,
            "semantic": semantic_result,
            "outputs": outputs,
            "corpus": bool(docs),
            "bridge_terms": bridge_terms,
            "ai_fallback": ai_fallback,
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"[{TOOL_NAME}] 找到 {len(segments)} 段：")
    for index, segment in enumerate(segments, 1):
        source = segment.get("source") or {}
        source_text = f"[{source.get('slug')}] " if source else ""
        print(f"  {index}. {source_text}完整讨论 "
              f"{segment['start_text']} – {segment['end_text']} "
              f"（{_fmt_duration(segment['duration'])}）")
        print(f"     核心片段 {segment['core_start_text']} – {segment['core_end_text']} "
              f"（{_fmt_duration(segment['core_duration'])}）")
        semantic = segment.get("semantic") or {}
        if semantic:
            print(f"     语义判定：{semantic.get('decision')} "
                  f"置信度 {semantic.get('confidence')}；{semantic.get('reason', '')}")
        print(f"     {segment['quote'][:120]}")
    if outputs:
        print(f"\n输出目录：{outputs['out_dir']}")
        print(f"报告：{outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
