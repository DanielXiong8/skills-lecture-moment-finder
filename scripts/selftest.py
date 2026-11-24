#!/usr/bin/env python3
"""Self-test: offline locator, live matcher and full live-span simulation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moment_core import build_outputs, parse_transcript, search  # noqa: E402
from live_core import Interest, LiveMatcher  # noqa: E402
from live_engine import Engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "examples" / "lecture-transcript.json"
QUERY = "老师讲动态规划状态转移方程的那几分钟"


def check_offline() -> dict:
    _meta, turns = parse_transcript(TRANSCRIPT)
    assert len(turns) > 10, "example transcript is too small"

    segments = search(turns, QUERY, top=3)
    assert segments, "no segment found for the example query"
    best = segments[0]
    assert "动态规划" in best["text"], "top span does not mention 动态规划"
    assert "状态转移方程" in best["text"], "top span does not mention 状态转移方程"
    assert 2360 <= best["start"] <= 2420, (
        f"expected the full DP discussion to start around 40:00, got {best['start_text']}"
    )
    assert best["end"] >= 2660, (
        f"expected the span to reach the exam-scope turn, got {best['end_text']}"
    )
    assert best["duration"] >= 180, (
        f"full discussion span is too short: {int(best['duration'])}s"
    )
    assert 30 <= best["core_duration"] <= 95, (
        f"core window outside 30-95s: {int(best['core_duration'])}s"
    )
    assert best["core_start"] <= 2520 <= best["core_end"], (
        f"core window misses the key quote: {best['core_start_text']}-{best['core_end_text']}"
    )
    assert "状态定义是骨架" in best["core_text"], (
        f"core text misses the key quote: {best['core_text'][:120]}"
    )

    with tempfile.TemporaryDirectory(prefix="lmf_selftest_") as temporary:
        outputs = build_outputs(
            Path(temporary) / "outputs",
            query=QUERY,
            transcript_path=TRANSCRIPT,
            segments=segments,
            notes=["selftest"],
        )
        out = Path(outputs["out_dir"])
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert result["query"] == QUERY, "result.json lost the query"
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "如何回放" in report, "report has no replay instructions"
        excerpt = (out / "transcript_excerpt.txt").read_text(encoding="utf-8")
        assert "状态转移方程" in excerpt, "transcript excerpt misses the topic"
        for forbidden in ("manifest.json", "SHA256SUMS.txt"):
            assert not (out / forbidden).exists(), (
                f"{forbidden} must not be produced by the learning skill"
            )
        assert not out.with_suffix(".zip").exists(), (
            "learning skill must not produce an evidence-pack zip"
        )
    return best


def check_live_matcher(turns) -> None:
    interests = [
        Interest(
            id="dp",
            question="老师讲动态规划状态转移方程的那几分钟",
            keywords=["动态规划", "状态转移方程"],
            min_score=15.0,
            cooldown_seconds=60,
        ),
        Interest(
            id="exam",
            question="期末考试范围",
            keywords=["考试范围", "考点"],
            min_score=12.0,
            cooldown_seconds=60,
        ),
    ]
    matcher = LiveMatcher(interests, window_seconds=20.0)
    live_hits = []
    for turn in turns:
        hit = matcher.update(turn.text, turn.end, turn.start)
        if hit:
            live_hits.append(hit)
    assert any(hit["interest_id"] == "dp" for hit in live_hits), (
        "live matcher did not fire for the DP interest"
    )
    assert any(hit["interest_id"] == "exam" for hit in live_hits), (
        "live matcher did not fire for the exam interest"
    )


def check_live_span() -> dict:
    """Run the real engine on the example lecture and require the full span."""
    with tempfile.TemporaryDirectory(prefix="lmf_live_selftest_") as temporary:
        root = Path(temporary)
        config = {
            "session_id": "selftest-live",
            "title": "selftest",
            "consent": True,
            "save_transcript": True,
            "interests": [
                {
                    "id": "dp",
                    "question": QUERY,
                    "keywords": ["动态规划", "状态转移方程"],
                    "min_score": 15.0,
                    "cooldown_seconds": 60,
                },
                {
                    "id": "exam",
                    "question": "期末考试范围",
                    "keywords": ["考试范围", "考点"],
                    "min_score": 12.0,
                    "cooldown_seconds": 60,
                },
            ],
            "semantic": {"enabled": False},
            "capture": {"source": "loopback", "screen": False},
            "asr": {"model": "none"},
            "trigger": {
                "window_seconds": 20.0,
                "ring_seconds": 300.0,
                "min_span": 45.0,
                "max_span": 600.0,
                "end_grace_seconds": 60.0,
                "rel_threshold": 0.15,
                "bridge": 2,
                "max_gap": 20.0,
            },
        }
        config_path = root / "interests.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        out = root / "session"
        engine = Engine(config_path, out, consent=True)
        assert engine.run_simulation(TRANSCRIPT, speed=0) == 0, "simulation failed"

        events = [
            json.loads(line) for line in
            (out / "hits.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(event.get("event") == "hit_open" for event in events), (
            "no hit_open event"
        )
        closed = [event for event in events if event.get("event") == "hit_close"]
        assert closed, "no hit_close event"
        dp_hits = [event for event in closed if event["interest_id"] == "dp"]
        assert dp_hits, "no closed DP hit"
        # The DP chapter is long and may be split by the max-span safety cap.
        # One of the chunks must cover the key 40:00-44:48 discussion.
        dp = next(
            (event for event in dp_hits
             if event["span_start"] <= 2400 and event["span_end"] >= 2660),
            None,
        )
        assert dp is not None, (
            "no DP chunk covers the 40:00-44:48 discussion: "
            + ", ".join(f"{event['span_start_text']}-{event['span_end_text']}"
                        for event in dp_hits)
        )

        # The lecture covers DP from 40:00 to 44:48. The live engine must save
        # that complete stretch, not a fixed 90-second lookback.
        assert 2360 <= dp["span_start"] <= 2420, (
            f"unexpected DP span start: {dp['span_start_text']}"
        )
        assert dp["span_end"] >= 2660, (
            f"DP span ended too early: {dp['span_end_text']}"
        )
        assert dp["duration"] >= 180, f"DP span too short: {dp['duration']}s"
        assert 30 <= dp["core_duration"] <= 95, (
            f"live core outside 30-95s: {dp['core_duration']}s"
        )
        assert "状态定义是骨架" in dp["core_text"], (
            f"live core misses the key quote: {dp['core_text'][:120]}"
        )

        hit_dir = Path(dp["dir"])
        assert (hit_dir / "hit.json").is_file(), "hit.json missing"
        assert (hit_dir / "transcript.txt").is_file(), "transcript.txt missing"
        for forbidden in ("manifest.json", "SHA256SUMS.txt"):
            assert not (hit_dir / forbidden).exists(), (
                f"{forbidden} must not be produced by the learning skill"
            )
        assert not hit_dir.with_suffix(".zip").exists(), (
            "learning skill must not produce an evidence-pack zip"
        )

        session = json.loads((out / "session.json").read_text(encoding="utf-8"))
        assert session["hits"] >= 1, "session.json has no hits"
        assert all("sha256" not in item for item in session["files"]), (
            "session.json must not carry hashes in the learning skill"
        )
    return dp


def check_live_semantic() -> dict:
    """Host judge path: request -> fake host response -> semantic hit -> close."""
    with tempfile.TemporaryDirectory(prefix="lmf_semantic_") as tmp:
        root = Path(tmp)
        config = {
            "session_id": "semantic-selftest",
            "consent": True,
            "interests": [{
                "id": "kp1",
                "question": "老师怎么解释第二型曲面积分的定向？",
                "definition": "用法向量和通量定义第二型曲面积分",
            }],
            "semantic": {
                "enabled": True, "backend": "host",
                "window_seconds": 45.0, "interval_seconds": 10.0,
                "min_new_chars": 1, "min_confidence": 0.6,
            },
            "trigger": {
                "min_span": 30.0, "max_span": 600.0,
                "semantic_grace_seconds": 90.0,
                "manual_span_seconds": 120.0,
            },
        }
        config_path = root / "interests.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        out = root / "session"
        engine = Engine(
            config_path, out, consent=True, semantic_override="host"
        )
        assert engine.matcher is None, "semantic mode must not use LiveMatcher"
        lines = [
            (0, 10, "今天讲第二型曲面积分"),
            (10, 20, "先看定向曲面和法向量"),
            (20, 30, "通量就是向量场穿过曲面的流量"),
            (30, 40, "所以定义成 P dydz 加 Q dzdx 加 R dxdy"),
            (40, 50, "这就是第二型曲面积分"),
            (50, 60, "下面看一个例题"),
            (60, 70, "例题结束"),
        ]
        for start, end, text in lines:
            engine._record_line(start, end, text)

        request_id = engine.semantic.maybe_request(engine.transcript_lines, 40)
        assert request_id, "semantic request was not created"
        request = json.loads(
            (out / "judge" / "requests" / f"{request_id}.json")
            .read_text(encoding="utf-8")
        )
        assert request["protocol"] == "lecture-live-judge/1"
        assert request["interests"][0]["id"] == "kp1"

        response = {"request_id": request_id, "judgments": [{
            "interest_id": "kp1", "level": "teach", "confidence": 0.95,
            "start": 5, "end": 35,
            "quote": "通量就是向量场穿过曲面的流量",
            "reason": "正在定义通量与第二型曲面积分",
        }]}
        (out / "judge" / "responses").mkdir(parents=True, exist_ok=True)
        (out / "judge" / "responses" / f"{request_id}.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8"
        )
        events = engine.semantic.poll()
        assert events, "host response was not polled"
        engine._apply_semantic(events, 3, None, None)
        assert "kp1" in engine.open_topics, "teach did not open a topic"
        topic = engine.open_topics["kp1"]
        assert topic.semantic and topic.manual is False
        assert topic.span_start == 5.0

        # Two negative windows close the topic; no keyword fallback involved.
        for seq, (window_start, window_end) in enumerate(
            [(30, 60), (40, 70)], start=2
        ):
            rid = f"req_{seq:04d}"
            (out / "judge" / "requests" / f"{rid}.json").write_text(
                json.dumps({
                    "protocol": "lecture-live-judge/1",
                    "request_id": rid,
                    "window": {"start": window_start, "end": window_end},
                    "interests": [{"id": "kp1", "question": "q"}],
                    "transcript": [],
                }, ensure_ascii=False), encoding="utf-8"
            )
            (out / "judge" / "responses" / f"{rid}.json").write_text(
                json.dumps({
                    "request_id": rid,
                    "judgments": [{
                        "interest_id": "kp1", "level": "none",
                        "confidence": 0.8, "start": window_start,
                        "end": window_end, "reason": "已转到例题",
                    }],
                }, ensure_ascii=False), encoding="utf-8"
            )
            engine._apply_semantic(
                engine.semantic.poll(), len(engine.transcript_lines) - 1,
                None, None,
            )
            engine._advance_topics(
                len(engine.transcript_lines) - 1, window_end, None, None
            )
        assert "kp1" not in engine.open_topics, "semantic topic did not close"

        # Manual VAR still works without any LLM response.
        (out / "MANUAL.json").write_text(
            json.dumps({"at": 80, "note": "这里没听懂"}, ensure_ascii=False),
            encoding="utf-8",
        )
        engine._check_manual(80, len(engine.transcript_lines) - 1, None, None)
        assert "manual" in engine.open_topics, "manual VAR did not open"
        engine._advance_topics(
            len(engine.transcript_lines) - 1, 201, None, None
        )
        assert "manual" not in engine.open_topics, "manual VAR did not close"

        events = [
            json.loads(line) for line in
            (out / "hits.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        closed = [item for item in events if item.get("event") == "hit_close"]
        semantic = [item for item in closed if item.get("trigger") == "semantic"]
        manual = [item for item in closed if item.get("trigger") == "manual"]
        assert semantic, "no semantic hit_close"
        assert manual, "no manual hit_close"
        assert semantic[0]["close_reason"] == "semantic_end"
        assert manual[0]["close_reason"] == "manual_end"
        assert manual[0]["span_end"] - manual[0]["span_start"] <= 121
        return semantic[0]


def main() -> int:
    _meta, turns = parse_transcript(TRANSCRIPT)
    best = check_offline()
    check_live_matcher(turns)
    dp = check_live_span()
    semantic = check_live_semantic()
    print("SELFTEST PASS")
    print(f"  offline full span: {best['start_text']} - {best['end_text']} "
          f"({int(best['duration'])}s)")
    print(f"  offline core:      {best['core_start_text']} - {best['core_end_text']} "
          f"({int(best['core_duration'])}s)")
    print(f"  live full span:    {dp['span_start_text']} - {dp['span_end_text']} "
          f"({int(dp['duration'])}s)")
    print(f"  live core:         {dp['core_start_text']} - {dp['core_end_text']} "
          f"({int(dp['core_duration'])}s)")
    print(f"  live quote: {dp['key_quote'][:80]}")
    print(f"  semantic hit: {semantic['span_start_text']} - "
          f"{semantic['span_end_text']} ({semantic['close_reason']})")
    print("  manual VAR: closed without an LLM backend")
    print("  no manifest/SHA256SUMS/zip in learning outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
