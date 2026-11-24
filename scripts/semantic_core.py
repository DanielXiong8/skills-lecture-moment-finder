"""Semantic hit rule for lecture-moment-finder.

Pipeline:
  1. Recall candidate windows with the existing keyword search plus windows
     around the knowledge point's `must_include` terms.
  2. Ask a local LLM whether each candidate actually *teaches* the point
     (definition/theorem/example), not merely mentions it.
  3. If no candidate passes, return "not found".
  4. Otherwise use the LLM's sentence range as the teaching window and pick a
     dense 30-90s core inside it.

The module is optional: when llama-cpp or the model is unavailable, callers
should fall back to `moment_core.search`.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moment_core import (
    Turn,
    _dense_core,
    _score_turn,
    fmt_ts,
    norm,
    parse_transcript,
    query_terms,
    render_window,
    search,
)

DEFAULT_MODEL = os.environ.get("LECTURE_SEMANTIC_MODEL", "")


@dataclass
class Candidate:
    start: float
    end: float
    turn_lo: int
    turn_hi: int
    keyword_score: float = 0.0
    coverage: float = 0.0
    present: list[str] = field(default_factory=list)
    text: str = ""
    source: str = ""
    source_date: str = ""
    source_part: str = ""
    origin: str = ""


@dataclass
class TranscriptDoc:
    slug: str
    date: str
    part: str
    path: Path
    turns: list[Turn]
    text: str = ""


SYNONYMS = {
    "柯西": ["科西"],
    "2^n": ["2的n次方", "二的n次方", "2n次方", "2 的 n 次方", "2的 n 次方"],
    "克莱罗": ["clairaut", "clearout", "clarrote"],
    "施瓦茨": ["schwarz", "shots", "she was"],
    "雅可比": ["jacobi", "结口比", "cocop", "接口比"],
    "海瑟": ["hessian", "hesion", "hashing", "heltion", "黑塞"],
    "富比尼": ["fubini", "负比尼"],
    "莱布尼茨": ["leibniz", "莱布尼兹"],
    "魏尔斯特拉斯": ["weierstrass"],
    "狄利克雷": ["dirichlet"],
    "阿贝尔": ["abel"],
    "拉格朗日": ["lagrange"],
    "斯托克斯": ["stokes", "stocks"],
    "格林": ["green"],
    "高斯": ["gauss"],
}


def _match_text(text: str) -> str:
    """Lowercase, strip spaces/punctuation, and fold common ASR variants."""
    value = norm(text)
    for canonical, variants in SYNONYMS.items():
        key = norm(canonical)
        for variant in variants:
            value = value.replace(norm(variant), key)
    return value


def split_groups(items: list[str]) -> list[list[str]]:
    groups = []
    for item in items:
        alternatives = [part.strip() for part in str(item).split("|") if part.strip()]
        if alternatives:
            groups.append(alternatives)
    return groups


def group_present(text: str, group: list[str]) -> bool:
    lowered = _match_text(text)
    return any(_match_text(alt) in lowered for alt in group)


def coverage(text: str, spec: dict) -> tuple[float, list[str]]:
    groups = split_groups(spec.get("must_include", []))
    if not groups:
        return 0.0, []
    present = [group[0] for group in groups if group_present(text, group)]
    return len(present) / len(groups), present


_STARTS_CACHE: dict[int, tuple[int, list[float]]] = {}


def _turn_starts(turns: list[Turn]) -> list[float]:
    key = id(turns)
    cached = _STARTS_CACHE.get(key)
    if cached is None or cached[0] != len(turns):
        starts = [turn.start for turn in turns]
        _STARTS_CACHE[key] = (len(turns), starts)
        return starts
    return cached[1]


def _candidate_from_range(turns: list[Turn], start: float, end: float, spec: dict) -> Candidate:
    starts = _turn_starts(turns)
    lo = max(0, bisect.bisect_left(starts, start) - 1)
    while lo < len(turns) and turns[lo].end <= start:
        lo += 1
    hi = bisect.bisect_left(starts, end)
    selected = [(index, turns[index]) for index in range(lo, hi)]
    if not selected:
        return Candidate(start, end, 0, 0)
    text = " ".join(t.text for _i, t in selected)
    cov, present = coverage(text, spec)
    return Candidate(
        start=selected[0][1].start,
        end=selected[-1][1].end,
        turn_lo=selected[0][0],
        turn_hi=selected[-1][0],
        coverage=cov,
        present=present,
        text=text,
    )


def _overlap_ratio(left: Candidate, right: Candidate) -> float:
    inter = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    shorter = min(left.end - left.start, right.end - right.start)
    return inter / shorter if shorter > 0 else 0.0


METHOD_CUES = (
    "定义", "定理", "命题", "证明", "推导", "公式", "设", "称", "求导",
    "换元", "变量代换", "链式法则", "雅可比", "解得", "例题", "例子",
)


def _bucket_select(candidates: list[Candidate], limit: int,
                   total_end: float = 0.0) -> list[Candidate]:
    """One candidate per equal time segment, then fill by score.

    A lecture can have dozens of near-duplicate hits in one region while the
    exact teaching segment sits 30 minutes later. Equal-time segments force
    the pool to cover the whole lecture; the LLM still decides which one is
    actually teaching.
    """
    if not candidates or limit <= 0:
        return []
    if total_end <= 0:
        total_end = max(candidate.end for candidate in candidates)
    if total_end <= 0:
        return candidates[:limit]

    def score(candidate: Candidate) -> float:
        return candidate.coverage * 100.0 + min(candidate.keyword_score, 60.0)

    # Fine dedupe first (3-minute buckets), keeping the best per bucket.
    fine: dict[int, Candidate] = {}
    for candidate in candidates:
        key = int(candidate.start // 180.0)
        old = fine.get(key)
        if old is None or score(candidate) > score(old):
            fine[key] = candidate
    pool = list(fine.values())
    if len(pool) <= limit:
        return sorted(pool, key=lambda c: (-score(c), c.start))[:limit]

    segment = total_end / float(limit)
    chosen: list[Candidate] = []
    for index in range(limit):
        lo = index * segment
        hi = (index + 1) * segment
        bucket = [c for c in pool if lo <= c.start < hi]
        if bucket:
            chosen.append(max(bucket, key=score))
    # Fill any empty segments with the best remaining candidates.
    chosen_ids = {id(item) for item in chosen}
    for candidate in sorted(pool, key=lambda c: (-score(c), c.start)):
        if len(chosen) >= limit:
            break
        if id(candidate) not in chosen_ids:
            chosen.append(candidate)
            chosen_ids.add(id(candidate))
    return sorted(chosen, key=lambda c: (-score(c), c.start))[:limit]


HOST_INSTRUCTIONS_V1 = """请对每个 candidate 判定它是否在“讲解”这个知识点。只输出 JSON，不要解释。

对每个 candidate 给出：
- level: "teach" | "mention" | "unrelated"
  - teach：定义、解释、证明、推导或完整例题说明该知识点；
  - mention：相关提及、方法列表、复习/考试范围、只讲到一半、或在别的证明/计算里顺带用到；
  - unrelated：无关或关键词巧合。
- start / end：相对 excerpt 的编号（[0]..），只包含真正在讲该知识点的连续条目；
- verify：level=teach 时，这段是否明确陈述了 definition 中的具体命题/条件/公式；
  只给结论名称、或在别的证明里用到，填 false；
- confidence：0.0–1.0；
- reason：一句中文。

输出格式：
{"judgments": [
  {"candidate_id": 1, "level": "teach|mention|unrelated",
   "start": 0, "end": 10, "verify": true, "confidence": 0.9, "reason": "..."}
]}"""


def candidate_windows(
    turns: list[Turn],
    spec: dict,
    max_candidates: int = 5,
    keyword_top: int = 15,
    bridge_terms: list[str] | None = None,
) -> list[Candidate]:
    """Recall candidates: keyword search + must-include / method / post-break windows.

    The old version only looked at +/-60s around a keyword hit. That missed two
    common cases: (a) the teacher announces the topic, takes a break, then
    explains it; (b) the explanation uses "对恒等式求导/链式法则" instead of
    repeating the topic name. Both now get their own candidate windows.
    """
    bridge_terms = [str(item) for item in (bridge_terms or spec.get("bridge_terms", []))
                    if str(item).strip()]
    query = spec.get("name", "") + " " + " ".join(
        item.replace("|", " ") for item in spec.get("must_include", [])
    )
    if bridge_terms:
        query += " " + " ".join(bridge_terms)
    candidates: list[Candidate] = []

    for segment in search(turns, query, top=keyword_top):
        candidate = _candidate_from_range(turns, segment["start"], segment["end"], spec)
        candidate.keyword_score = float(segment.get("raw_score", 0.0))
        candidates.append(candidate)

    groups = split_groups(spec.get("must_include", []))
    # Normalize once per turn/group. The old per-call normalization was the
    # dominant cost when scanning a whole course.
    group_norm = [
        [_match_text(alt) for alt in group] for group in groups
    ]
    turn_norm = [_match_text(turn.text) for turn in turns]
    matching_turns = [
        index for index, text in enumerate(turn_norm)
        if any(any(alt in text for alt in group) for group in group_norm)
    ]
    window_hits = matching_turns
    if len(window_hits) > 250:
        step = max(1, len(window_hits) // 250)
        window_hits = window_hits[::step][:250]

    for index in window_hits:
        start = max(0.0, turns[index].start - 60.0)
        end = min(turns[-1].end, turns[index].end + 90.0)
        candidates.append(_candidate_from_range(turns, start, end, spec))

        # Forward explanation window: a term hit often opens a 2-3 minute
        # derivation. The symmetric window cuts it off halfway.
        forward_start = max(0.0, turns[index].start - 60.0)
        forward_end = min(turns[-1].end, turns[index].start + 180.0)
        if forward_end - forward_start > 120.0:
            candidate = _candidate_from_range(
                turns, forward_start, forward_end, spec
            )
            candidate.keyword_score = max(candidate.keyword_score, 30.0)
            candidates.append(candidate)

        # Continue after a long pause/break: the actual explanation often
        # starts in the next segment, not before the break.
        probe = index + 1
        while probe < len(turns) and turns[probe].start - turns[index].start <= 240.0:
            gap = turns[probe].start - turns[probe - 1].end
            if gap >= 20.0:
                start = turns[probe].start
                end = min(turns[-1].end, start + 180.0)
                candidate = _candidate_from_range(turns, start, end, spec)
                # Keep this window through overlap merging: the explanation
                # after a break is often the actual teaching segment.
                candidate.keyword_score = max(candidate.keyword_score, 20.0)
                candidates.append(candidate)
                break
            probe += 1

    # Method/definition sentences near a term hit. This is what catches
    # "differentiate the identity, use the chain rule, solve the system".
    for index, turn in enumerate(turns):
        if not any(cue in turn.text for cue in METHOD_CUES):
            continue
        pos = bisect.bisect_left(matching_turns, index)
        near = False
        for probe in (pos - 1, pos):
            if 0 <= probe < len(matching_turns):
                if abs(matching_turns[probe] - index) <= 6:
                    near = True
                    break
        if not near:
            continue
        start = max(0.0, turn.start - 45.0)
        end = min(turns[-1].end, turn.end + 75.0)
        candidates.append(_candidate_from_range(turns, start, end, spec))

    if bridge_terms:
        for index, turn in enumerate(turns):
            if any(term.casefold() in turn.text.casefold() for term in bridge_terms):
                start = max(0.0, turn.start - 60.0)
                end = min(turns[-1].end, turn.end + 90.0)
                candidates.append(_candidate_from_range(turns, start, end, spec))

    # merge overlapping windows, keeping the one with better coverage/score
    candidates.sort(key=lambda c: (-c.coverage, -c.keyword_score, -(c.end - c.start)))
    merged: list[Candidate] = []
    for candidate in candidates:
        hit = next(
            (index for index, kept in enumerate(merged)
             if _overlap_ratio(candidate, kept) >= 0.6),
            None,
        )
        if hit is None:
            merged.append(candidate)
            continue
        kept = merged[hit]
        start = min(kept.start, candidate.start)
        end = max(kept.end, candidate.end)
        if end - start <= 300.0:
            union = _candidate_from_range(turns, start, end, spec)
            union.keyword_score = max(kept.keyword_score, candidate.keyword_score)
            merged[hit] = union
        elif (candidate.coverage, candidate.keyword_score) > (
                kept.coverage, kept.keyword_score):
            merged[hit] = candidate

    # Coverage first: a window that touches all must-include groups is more
    # likely to be the teaching segment than a high-keyword mention window.
    merged.sort(
        key=lambda c: (-(c.coverage * 100.0 + min(c.keyword_score, 60.0)), -c.coverage)
    )
    selected = _bucket_select(merged, max_candidates, turns[-1].end)

    # Recall guarantee for contract terms: make sure the LLM sees at least a
    # couple of windows that explicitly contain a must-include term, even if
    # their keyword score is lower. The LLM still makes the final choice.
    contract_terms = [
        max(group, key=len) for group in groups if len(max(group, key=len)) >= 2
    ]
    if contract_terms and len(selected) >= max_candidates:
        extras = []
        for candidate in sorted(
            merged,
            key=lambda c: (-(c.coverage * 100.0 + min(c.keyword_score, 60.0)), c.start),
        ):
            text = _match_text(candidate.text)
            if any(_match_text(term) in text for term in contract_terms):
                if candidate not in selected:
                    extras.append(candidate)
                if len(extras) >= 2:
                    break
        selected = sorted(
            selected + extras,
            key=lambda c: (-(c.coverage * 100.0 + min(c.keyword_score, 60.0)), c.start),
        )[:max_candidates]
    return selected


def build_excerpt(
    turns: list[Turn],
    start: float,
    end: float,
    pad: float = 20.0,
    max_sec: float = 180.0,
) -> tuple[list[tuple[int, Turn]], str]:
    lo = max(0.0, start - pad)
    hi = min(turns[-1].end, end + pad)
    if hi - lo > max_sec:
        center = (start + end) / 2.0
        lo = max(0.0, center - max_sec / 2.0)
        hi = min(turns[-1].end, center + max_sec / 2.0)
    selected = [(i, t) for i, t in enumerate(turns) if t.end > lo and t.start < hi]
    lines = [f"[{local}] {t.start:.0f}s {t.text}" for local, (_i, t) in enumerate(selected)]
    return selected, "\n".join(lines)


def verify_evidence(payload: dict, excerpt: str, spec: dict) -> tuple[float, list[str]]:
    """Check that the LLM's quoted evidence really appears in the excerpt."""
    evidence = payload.get("evidence") or {}
    if isinstance(evidence, list):
        evidence = {str(i): item for i, item in enumerate(evidence)}
    if not isinstance(evidence, dict):
        evidence = {}
    groups = split_groups(spec.get("must_include", []))
    if not groups:
        return 1.0, []
    normalized_excerpt = _match_text(excerpt)
    verified = 0
    missing = []
    for group in groups:
        quote = None
        for key, value in evidence.items():
            key_text = str(key)
            if any(alt in key_text or key_text in alt for alt in group):
                quote = value
                break
        if not isinstance(quote, str):
            missing.append(group[0])
            continue
        quote = re.sub(r"【[^】]*】", "", quote)
        quote = re.sub(r"^\s*\d+(?:\.\d+)?s?\s*", "", quote)
        quote = re.sub(r"[.…]+", "", quote).strip()
        if quote.lower() in {"null", "none", "无", "没有", ""}:
            missing.append(group[0])
            continue
        # the quoted evidence itself must mention the group's concept
        if not group_present(quote, group):
            missing.append(group[0])
            continue
        if normalized_excerpt and _match_text(quote) and _match_text(quote) in normalized_excerpt:
            verified += 1
        else:
            missing.append(group[0])
    return verified / len(groups), missing


def _parse_json(text: str) -> dict:
    text = text or ""
    text = re.sub(r"^.*?<｜end▁of▁thinking｜>", "", text, flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class SemanticJudge:
    """Lazy local-LLM judge using llama-cpp-python."""

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path or DEFAULT_MODEL
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self.error = ""
        self.calls = 0

    @property
    def available(self) -> bool:
        if self._llm is not None:
            return True
        if not self.model_path:
            self.error = ("未配置本地语义模型；用 --model 或 LECTURE_SEMANTIC_MODEL，"
                          "或使用 --semantic host（推荐）")
            return False
        if not Path(self.model_path).is_file():
            self.error = f"semantic model not found: {self.model_path}"
            return False
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            self.error = "llama-cpp-python is not installed"
            return False
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def judge(self, spec: dict, excerpt: str) -> dict:
        if not self.available:
            return {"teach": False, "confidence": 0.0, "reason": self.error, "_raw": ""}
        groups = spec.get("must_include", [])
        prompt = f"""你是课堂助教。判断一段课堂转写是否在“讲解”指定知识点，而不是仅仅提到，或用相关术语去证明别的结论。

知识点：{spec.get('name', '')}
具体内容（必须是在讲这个命题/定义/定理，而不是别的）：{spec.get('definition', '')}
必须讲到的证据项：{'、'.join(groups)}
只算提及：{spec.get('mention_only', '只在考试范围、复习清单、或顺带一提不算。')}

课堂转写（[编号] 秒数 文本）：
{excerpt}

请分两步，只输出 JSON，不要解释：
第一步：为每个“证据项”在原文中找逐字证据，写入 evidence；找不到就写 null。不要改写原文。
第二步：把这段分成三类之一：
- "teach"：老师在定义、解释、证明、推导或完整例题说明这个具体内容；
- "mention"：相关提及、方法列表、复习/考试范围、只讲到一半、或只是用相关术语；
- "unrelated"：无关或关键词巧合。
输出：
{{"evidence": {{"证据项1": "原文或null", "证据项2": "原文或null"}}, "level": "teach|mention|unrelated", "start": 起始编号, "end": 结束编号, "confidence": 0.0 到 1.0, "reason": "一句中文"}}

规则：
- 只有老师真正在定义、解释、证明、推导或完整例题说明这个具体内容时，level="teach"。
- 只出现术语、考试范围、复习清单、顺带一提、方法列表、或只讲到一半，level="mention"。
- 用距离/三角不等式等术语证明别的结论，或者完全无关，level="unrelated"。
- start/end 只包含真正在讲这个知识点的连续条目，不要包含无关前后文。
/no_think"""
        started = time.time()
        payload = {}
        raw = ""
        for _attempt in range(2):
            try:
                out = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "你只输出 JSON，不要解释。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=600,
                    temperature=0,
                )
                raw = out["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                return {"teach": False, "confidence": 0.0,
                        "reason": f"judge failed: {type(exc).__name__}: {exc}", "_raw": ""}
            payload = _parse_json(raw)
            if payload:
                break
        self.calls += 1
        if not payload:
            payload = {"level": "unrelated", "confidence": 0.0,
                       "reason": "judge returned invalid/truncated JSON"}
        level = str(payload.get("level") or "").strip().lower()
        if level not in {"teach", "mention", "unrelated"}:
            level = "teach" if payload.get("teach") else "unrelated"
        payload["level"] = level
        payload["teach"] = level == "teach"
        payload["mention"] = level == "mention"
        verified, missing = verify_evidence(payload, excerpt, spec)
        payload["_evidence_verified"] = round(verified, 3)
        payload["_evidence_missing"] = missing
        payload["_raw"] = raw[:300]
        payload["_seconds"] = round(time.time() - started, 2)
        return payload


    def verify(self, spec: dict, excerpt: str) -> dict:
        """Focused second pass: is the excerpt stating this exact claim?"""
        if not self.available:
            return {"confirm": False, "confidence": 0.0, "reason": self.error}
        prompt = f"""你是课堂助教。只判断一件事：这段课堂转写是否在陈述、证明或定义下面这个具体命题。

知识点：{spec.get('name', '')}
具体命题：{spec.get('definition', '')}
只算提及：{spec.get('mention_only', '只在考试范围、复习清单、或顺带一提不算。')}

课堂转写：
{excerpt[:3200]}

如果这段只是在用相关术语证明别的结论，或者只提到知识点名称，回答 false。
如果这段只是在一个具体计算/证明中“用到”该知识点，但没有陈述它的定义/条件/结论/公式，回答 false。
如果只给出结论名称（例如“存在且 C1”）而没有给出关键条件/公式，回答 false。
请引用一句原文作为证据（不要带编号/时间戳）。
只输出 JSON：{{"confirm": true 或 false, "quote": "原文或空", "confidence": 0.0 到 1.0, "reason": "一句中文"}}
/no_think"""
        started = time.time()
        try:
            out = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0,
            )
            raw = out["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return {"confirm": False, "confidence": 0.0,
                    "reason": f"verify failed: {type(exc).__name__}: {exc}"}
        self.calls += 1
        payload = _parse_json(raw)
        payload["_seconds"] = round(time.time() - started, 2)
        payload["_raw"] = raw[:300]
        return payload


def _expand_window(
    turns: list[Turn],
    lo: int,
    hi: int,
    spec: dict,
    min_span: float = 30.0,
    max_span: float = 600.0,
) -> tuple[int, int]:
    groups = split_groups(spec.get("must_include", []))
    lo = max(0, lo)
    hi = min(len(turns) - 1, hi)
    # expand for short windows
    while turns[hi].end - turns[lo].start < min_span:
        can_left = lo > 0
        can_right = hi < len(turns) - 1
        if not can_left and not can_right:
            break
        if can_left:
            lo -= 1
        if can_right:
            hi += 1
    # include one adjacent context sentence if it is close or on-topic
    for side in ("left", "right"):
        if side == "left" and lo > 0:
            gap = turns[lo].start - turns[lo - 1].end
            if gap <= 5.0 or any(group_present(turns[lo - 1].text, g) for g in groups):
                lo -= 1
        if side == "right" and hi < len(turns) - 1:
            gap = turns[hi + 1].start - turns[hi].end
            if gap <= 5.0 or any(group_present(turns[hi + 1].text, g) for g in groups):
                hi += 1
    # cap
    while turns[hi].end - turns[lo].start > max_span and hi > lo:
        center = (turns[lo].start + turns[hi].end) / 2.0
        if center - turns[lo].start >= turns[hi].end - center:
            lo += 1
        else:
            hi -= 1
    return lo, hi


def _semantic_core(
    turns: list[Turn],
    lo: int,
    hi: int,
    spec: dict,
    core_min: float = 30.0,
    core_max: float = 90.0,
) -> tuple[int, int]:
    best = (lo, hi)
    best_score = -1.0
    for start in range(lo, hi + 1):
        for end in range(start, hi + 1):
            duration = turns[end].end - turns[start].start
            if duration < core_min:
                continue
            if duration > core_max:
                break
            text = " ".join(turn.text for turn in turns[start:end + 1])
            cov, _present = coverage(text, spec)
            density = cov * 10.0 + len(text) / 500.0 - abs(duration - 60.0) / 600.0
            if density > best_score:
                best_score = density
                best = (start, end)
    return best


def _build_segment(
    turns: list[Turn],
    selected: list[tuple[int, Turn]],
    payload: dict,
    spec: dict,
    candidate: Candidate,
    rank: int,
    candidates_judged: int,
    decision: str,
    core_min: float,
    core_max: float,
) -> dict:
    local_start = payload.get("start")
    local_end = payload.get("end")
    if not isinstance(local_start, int) or not isinstance(local_end, int):
        local_start, local_end = 0, len(selected) - 1
    local_start = max(0, min(local_start, len(selected) - 1))
    local_end = max(local_start, min(local_end, len(selected) - 1))
    lo = selected[local_start][0]
    hi = selected[local_end][0]
    lo, hi = _expand_window(turns, lo, hi, spec)
    core_lo, core_hi = _semantic_core(turns, lo, hi, spec, core_min, core_max)

    groups = split_groups(spec.get("must_include", []))
    why = []
    for group in groups:
        if any(group_present(turn.text, group) for turn in turns[lo:hi + 1]):
            why.append(group[0])
    quote = ""
    for turn in turns[lo:hi + 1]:
        if any(group_present(turn.text, group) for group in groups):
            quote = turn.text
            break
    if not quote:
        quote = turns[lo].text

    confidence = float(payload.get("confidence") or 0.0)
    return {
        "start": round(turns[lo].start, 2),
        "end": round(turns[hi].end, 2),
        "duration": round(turns[hi].end - turns[lo].start, 2),
        "start_text": fmt_ts(turns[lo].start),
        "end_text": fmt_ts(turns[hi].end),
        "core_start": round(turns[core_lo].start, 2),
        "core_end": round(turns[core_hi].end, 2),
        "core_duration": round(turns[core_hi].end - turns[core_lo].start, 2),
        "core_start_text": fmt_ts(turns[core_lo].start),
        "core_end_text": fmt_ts(turns[core_hi].end),
        "turn_range": [lo, hi],
        "core_turn_range": [core_lo, core_hi],
        "speaker": turns[lo].speaker,
        "text": render_window(turns, turns[lo].start, turns[hi].end),
        "core_text": render_window(turns, turns[core_lo].start, turns[core_hi].end),
        "quote": quote,
        "score": round(confidence, 4),
        "raw_score": round(confidence, 4),
        "why": why[:4],
        "semantic": {
            "decision": decision,
            "level": payload.get("level"),
            "confidence": confidence,
            "reason": str(payload.get("reason") or ""),
            "candidate_rank": rank,
            "coverage": round(candidate.coverage, 3),
            "candidates_judged": candidates_judged,
            "evidence_verified": payload.get("_evidence_verified"),
            "verify": payload.get("_verify"),
        },
    }


def semantic_search(
    turns: list[Turn],
    spec: dict,
    judge: SemanticJudge,
    max_candidates: int = 5,
    min_coverage: float = 0.5,
    min_confidence: float = 0.6,
    min_verify_confidence: float = 0.6,
    core_min: float = 30.0,
    core_max: float = 90.0,
) -> dict:
    """Return {'segments': [...], 'decision': ..., 'reason': ..., 'candidates': [...]}."""
    if not judge.available:
        return {"segments": [], "decision": "unavailable", "reason": judge.error, "candidates": []}

    candidates = candidate_windows(turns, spec, max_candidates=max_candidates)
    candidates = [c for c in candidates if c.coverage >= min_coverage]
    if not candidates:
        return {
            "segments": [],
            "decision": "not_found",
            "reason": "没有候选覆盖足够多的必含项，语义规则直接判为未讲。",
            "candidates": [],
        }

    judged = []
    teach_segments = []
    mention_segments = []
    for rank, candidate in enumerate(candidates, 1):
        selected, excerpt = build_excerpt(turns, candidate.start, candidate.end)
        payload = judge.judge(spec, excerpt)
        level = str(payload.get("level") or "unrelated")
        confidence = float(payload.get("confidence") or 0.0)
        verify = None

        if level == "teach" and confidence >= min_confidence:
            verify = judge.verify(spec, excerpt)
            payload["_verify"] = verify
            verify_confidence = float(verify.get("confidence") or 0.0)
            if not (verify.get("confirm") and verify_confidence >= min_verify_confidence):
                level = "mention"
        if confidence < min_confidence and level == "teach":
            level = "mention"

        judged.append({
            "rank": rank,
            "start": candidate.start,
            "end": candidate.end,
            "start_text": fmt_ts(candidate.start),
            "end_text": fmt_ts(candidate.end),
            "coverage": round(candidate.coverage, 3),
            "keyword_score": round(candidate.keyword_score, 3),
            "level": level,
            "teach": level == "teach",
            "mention": level == "mention",
            "confidence": confidence,
            "evidence_verified": payload.get("_evidence_verified"),
            "evidence_missing": payload.get("_evidence_missing", []),
            "verify": verify,
            "reason": str(payload.get("reason") or ""),
            "seconds": payload.get("_seconds"),
            "local_start": payload.get("start"),
            "local_end": payload.get("end"),
        })
        if level not in {"teach", "mention"}:
            continue
        segment = _build_segment(
            turns, selected, payload, spec, candidate, rank, len(candidates),
            "teach" if level == "teach" else "mention_only",
            core_min, core_max,
        )
        if level == "teach":
            teach_segments.append(segment)
        else:
            mention_segments.append(segment)

    if teach_segments:
        best = max(teach_segments, key=lambda s: s["semantic"]["confidence"])
        return {"segments": [best], "decision": "teach",
                "reason": best["semantic"]["reason"], "candidates": judged}
    if mention_segments:
        best = max(mention_segments, key=lambda s: s["semantic"]["confidence"])
        best["semantic"]["reason"] = (
            "只找到相关提及/未完整展开，未确认是在完整讲解："
            + str(best["semantic"].get("reason") or "")
        )
        return {"segments": [best], "decision": "mention_only",
                "reason": best["semantic"]["reason"], "candidates": judged}
    return {
        "segments": [],
        "decision": "not_found",
        "reason": "所有候选均被语义判定为无关或未讲，未找到教学段。",
        "candidates": judged,
    }


HOST_INSTRUCTIONS = """请对每个 candidate 判定它是否在“讲解”这个知识点，并由你选出最贴合的一个。
只输出 JSON，不要解释。

对每个 candidate 给出：
- level: "teach" | "mention" | "unrelated"
  - teach：定义、解释、证明、推导或完整例题说明该知识点；
  - mention：相关提及、方法列表、复习/考试范围、只讲到一半、或在别的证明/计算里顺带用到；
  - unrelated：无关或关键词巧合。
- start / end：相对 excerpt 的编号（[0]..），只包含真正在讲该知识点的连续条目；
- verify：level=teach 时，这段是否明确陈述了 definition 中的具体命题/条件/公式；
  只给结论名称、或在别的证明里用到，填 false；
- confidence：0.0–1.0；
- reason：一句中文。

判定完全部 candidate 后，再做一次选择：
- best_candidate_id：最贴合知识点名称和 definition 的那个 candidate；
  注意限定词，例如“二重积分”不能被“三重积分”顶替，“可导性”不能被“连续性”顶替。
- selected_candidate_ids：如果一个 candidate 只覆盖定义的一部分，可以选多个
  （同一来源、时间相邻的会被合并）；否则只填 best_candidate_id。
- selection_reason：为什么选它，明确说明它如何匹配限定词。

若全部 candidate 都不构成 teach（例如转写把术语写错、候选池没罩到讲解段），
可额外输出 retry_terms：一个字符串数组，给出该知识点在口语/板书/ASR 里可能的
同义或错写表述（如“高斯公式”被转成“公因公式”），供第二轮检索重建候选池。
retry_terms 只用于检索，不要用它编造区间；第二轮仍由你判定 teach/mention。

输出格式：
{"best_candidate_id": 2, "selected_candidate_ids": [2],
 "selection_reason": "候选2是二重积分极坐标换元；候选1是三重积分",
 "judgments": [
  {"candidate_id": 1, "level": "teach|mention|unrelated",
   "start": 0, "end": 10, "verify": true, "confidence": 0.9, "reason": "..."}
]}"""


def prepare_host_request(
    turns: list[Turn],
    spec: dict,
    max_candidates: int = 12,
    bridge_terms: list[str] | None = None,
) -> dict:
    """Deterministic step: recall candidates and emit a host-agent judging request."""
    candidates = candidate_windows(
        turns, spec, max_candidates=max_candidates, bridge_terms=bridge_terms
    )
    items = []
    for rank, candidate in enumerate(candidates, 1):
        _selected, excerpt = build_excerpt(turns, candidate.start, candidate.end)
        items.append({
            "candidate_id": rank,
            "start": round(candidate.start, 2),
            "end": round(candidate.end, 2),
            "start_text": fmt_ts(candidate.start),
            "end_text": fmt_ts(candidate.end),
            "coverage": round(candidate.coverage, 3),
            "keyword_score": round(candidate.keyword_score, 3),
            "excerpt": excerpt,
        })
    return {
        "knowledge_point": {
            "id": spec.get("id", ""),
            "name": spec.get("name", ""),
            "definition": spec.get("definition", ""),
            "must_include": spec.get("must_include", []),
            "mention_only": spec.get("mention_only", ""),
            "bridge_terms": list(bridge_terms or spec.get("bridge_terms", [])),
        },
        "instructions": HOST_INSTRUCTIONS,
        "candidates": items,
    }


def _response_selected_ids(response: dict) -> list[int]:
    """Candidate ids explicitly chosen by the host model."""
    ids: list[int] = []
    raw = response.get("selected_candidate_ids")
    if isinstance(raw, list):
        for item in raw:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
    try:
        best = int(response.get("best_candidate_id"))
    except (TypeError, ValueError):
        best = None
    if best is not None and best not in ids:
        ids.append(best)
    return ids


def _select_segment(segments: list[dict], response: dict) -> dict:
    """Prefer the host model's explicit selection over script-side confidence."""
    selected = _response_selected_ids(response)
    for segment in segments:
        rank = segment.get("semantic", {}).get("candidate_rank")
        if rank in selected:
            return segment
    return max(segments, key=lambda item: item["semantic"]["confidence"])


def apply_host_response(
    turns: list[Turn],
    spec: dict,
    request: dict,
    response: dict,
    core_min: float = 30.0,
    core_max: float = 90.0,
) -> dict:
    """Apply the host agent's judgments and build the final teaching window."""
    candidates = {int(item["candidate_id"]): item for item in request.get("candidates", [])}
    judgments = response.get("judgments") or []
    teach_segments = []
    mention_segments = []
    judged = []

    for judgment in judgments:
        try:
            candidate_id = int(judgment.get("candidate_id"))
        except (TypeError, ValueError):
            continue
        candidate_data = candidates.get(candidate_id)
        if not candidate_data:
            continue
        level = str(judgment.get("level") or "unrelated").strip().lower()
        if level not in {"teach", "mention", "unrelated"}:
            level = "unrelated"
        verify = judgment.get("verify")
        if level == "teach" and verify is False:
            level = "mention"
        confidence = float(judgment.get("confidence") or 0.0)
        payload = {
            "level": level,
            "confidence": confidence,
            "reason": str(judgment.get("reason") or ""),
            "start": judgment.get("start"),
            "end": judgment.get("end"),
            "_verify": judgment,
            "_evidence_verified": judgment.get("evidence_verified"),
        }
        judged.append({
            "rank": candidate_id,
            "start": candidate_data["start"],
            "end": candidate_data["end"],
            "start_text": candidate_data["start_text"],
            "end_text": candidate_data["end_text"],
            "coverage": candidate_data.get("coverage"),
            "level": level,
            "teach": level == "teach",
            "mention": level == "mention",
            "confidence": confidence,
            "verify": judgment,
            "reason": payload["reason"],
        })
        if level not in {"teach", "mention"}:
            continue

        selected, _excerpt = build_excerpt(
            turns, float(candidate_data["start"]), float(candidate_data["end"]))
        if not selected:
            continue
        segment = _build_segment(
            turns, selected, payload, spec,
            Candidate(float(candidate_data["start"]), float(candidate_data["end"]),
                      selected[0][0], selected[-1][0]),
            candidate_id, len(candidates),
            "teach" if level == "teach" else "mention_only",
            core_min, core_max,
        )
        if level == "teach":
            teach_segments.append(segment)
        else:
            mention_segments.append(segment)

    if teach_segments:
        best = _select_segment(teach_segments, response)
        return {"segments": [best], "decision": "teach",
                "reason": best["semantic"]["reason"], "candidates": judged,
                "selection_reason": str(response.get("selection_reason") or "")}
    if mention_segments:
        best = _select_segment(mention_segments, response)
        best["semantic"]["reason"] = (
            "只找到相关提及/未完整展开，未确认是在完整讲解："
            + str(best["semantic"].get("reason") or "")
        )
        return {"segments": [best], "decision": "mention_only",
                "reason": best["semantic"]["reason"], "candidates": judged,
                "selection_reason": str(response.get("selection_reason") or "")}
    return {
        "segments": [],
        "decision": "not_found",
        "reason": "宿主判定所有候选均为提及/无关，未找到教学段。",
        "candidates": judged,
    }


def load_transcript_dir(path: str | Path, pattern: str = "*.json") -> list[TranscriptDoc]:
    """Load a course folder as multiple transcripts (one lecture per file)."""
    docs: list[TranscriptDoc] = []
    for file_path in sorted(Path(path).glob(pattern)):
        try:
            _meta, turns = parse_transcript(file_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not turns:
            continue
        data = {}
        if file_path.suffix.lower() == ".json":
            try:
                data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                data = {}
        docs.append(TranscriptDoc(
            slug=file_path.stem,
            date=str(data.get("date") or ""),
            part=str(data.get("part") or ""),
            path=file_path,
            turns=turns,
            text=" ".join(turn.text for turn in turns),
        ))
    if not docs:
        raise ValueError(f"目录里没有可用转写：{path}")
    return docs


def legacy_candidate_windows(
    turns: list[Turn],
    spec: dict,
    max_candidates: int = 6,
    keyword_top: int = 10,
) -> list[Candidate]:
    """The v1 single-lecture candidate generator, kept as a recall anchor.

    The cross-lecture generator can outrank the segment the old pipeline
    already found. Running both and letting the LLM choose is the union
    strategy that avoids throwing away known-good teaching windows.
    """
    query = spec.get("name", "") + " " + " ".join(
        item.replace("|", " ") for item in spec.get("must_include", [])
    )
    candidates: list[Candidate] = []
    for segment in search(turns, query, top=keyword_top):
        candidate = _candidate_from_range(turns, segment["start"], segment["end"], spec)
        candidate.keyword_score = float(segment.get("raw_score", 0.0))
        candidates.append(candidate)

    groups = split_groups(spec.get("must_include", []))
    group_norm = [[_match_text(alt) for alt in group] for group in groups]
    for index, turn in enumerate(turns):
        text = _match_text(turn.text)
        if not any(any(alt in text for alt in group) for group in group_norm):
            continue
        start = max(0.0, turn.start - 60.0)
        end = min(turns[-1].end, turn.end + 60.0)
        candidates.append(_candidate_from_range(turns, start, end, spec))

    candidates.sort(
        key=lambda c: (-c.coverage, -c.keyword_score, -(c.end - c.start))
    )
    merged: list[Candidate] = []
    for candidate in candidates:
        if any(_overlap_ratio(candidate, kept) >= 0.6 for kept in merged):
            continue
        merged.append(candidate)
    merged.sort(
        key=lambda c: (-(c.coverage * 20.0 + min(c.keyword_score, 60.0)), -c.coverage)
    )
    return merged[:max_candidates]



def legacy_candidate_windows_exact(
    turns: list[Turn],
    spec: dict,
    max_candidates: int = 6,
    keyword_top: int = 10,
    bridge_terms: list[str] | None = None,
) -> list[Candidate]:
    """v1 candidate generator; bridge_terms (host retry) only widen retrieval."""
    bridge_terms = list(bridge_terms or [])
    query = spec.get("name", "") + " " + " ".join(
        item.replace("|", " ") for item in spec.get("must_include", [])
    )
    if bridge_terms:
        query += " " + " ".join(bridge_terms)
    candidates: list[Candidate] = []

    for segment in search(turns, query, top=keyword_top):
        candidate = _candidate_from_range(turns, segment["start"], segment["end"], spec)
        candidate.keyword_score = float(segment.get("raw_score", 0.0))
        candidates.append(candidate)

    groups = split_groups(list(spec.get("must_include", [])) + bridge_terms)
    matching_turns = []
    for index, turn in enumerate(turns):
        if any(group_present(turn.text, group) for group in groups):
            matching_turns.append(index)
    for index in matching_turns:
        start = max(0.0, turns[index].start - 60.0)
        end = min(turns[-1].end, turns[index].end + 60.0)
        candidates.append(_candidate_from_range(turns, start, end, spec))

    candidates.sort(key=lambda c: (-c.coverage, -c.keyword_score, -(c.end - c.start)))
    merged: list[Candidate] = []
    for candidate in candidates:
        if any(_overlap_ratio(candidate, kept) >= 0.6 for kept in merged):
            continue
        merged.append(candidate)

    merged.sort(
        key=lambda c: (-(c.coverage * 20.0 + min(c.keyword_score, 60.0)), -c.coverage)
    )
    return merged[:max_candidates]


def prepare_host_request_legacy(
    turns: list[Turn],
    spec: dict,
    max_candidates: int = 6,
    bridge_terms: list[str] | None = None,
) -> dict:
    """v1 single-lecture request; bridge_terms enables the host retry round."""
    candidates = legacy_candidate_windows_exact(
        turns, spec, max_candidates=max_candidates, bridge_terms=bridge_terms
    )
    items = []
    for rank, candidate in enumerate(candidates, 1):
        _selected, excerpt = build_excerpt(turns, candidate.start, candidate.end)
        items.append({
            "candidate_id": rank,
            "start": round(candidate.start, 2),
            "end": round(candidate.end, 2),
            "start_text": fmt_ts(candidate.start),
            "end_text": fmt_ts(candidate.end),
            "coverage": round(candidate.coverage, 3),
            "keyword_score": round(candidate.keyword_score, 3),
            "excerpt": excerpt,
        })
    return {
        "knowledge_point": {
            "id": spec.get("id", ""),
            "name": spec.get("name", ""),
            "definition": spec.get("definition", ""),
            "must_include": spec.get("must_include", []),
            "mention_only": spec.get("mention_only", ""),
        },
        "instructions": HOST_INSTRUCTIONS,
        "candidates": items,
    }


def candidate_windows_corpus(
    docs: list[TranscriptDoc],
    spec: dict,
    max_candidates: int = 300,
    per_doc: int = 12,
    min_coverage: float = 0.0,
    bridge_terms: list[str] | None = None,
) -> list[Candidate]:
    """Recall candidates across every lecture, not just one primary lecture.

    `per_doc` keeps the pool diverse; `max_candidates` is the global budget.
    The old evaluation picked one lecture per knowledge point, which made a
    whole class of misses impossible to recover.
    """
    groups = split_groups(spec.get("must_include", []))
    bridge_terms = [str(item) for item in (bridge_terms or spec.get("bridge_terms", []))
                    if str(item).strip()]
    pool: list[Candidate] = []
    legacy_by_doc: dict[str, list[Candidate]] = {}
    new_by_doc: dict[str, list[Candidate]] = {}
    doc_scores: dict[str, float] = {}
    for doc in docs:
        # Cheap prefilter: require at least half of the must-include groups (or
        # a bridge term). A lecture with only one generic word such as "形式"
        # cannot teach the point, and scanning it is what made a whole-course
        # search slow.
        if groups:
            doc_norm = _match_text(doc.text)
            # Prefer the longest alternative in each group: "微分形式" is
            # specific, "形式" alone matches almost every lecture.
            specific = [
                max(group, key=lambda item: len(_match_text(item)))
                for group in groups
            ]
            specific_hits = sum(
                1 for term in specific if _match_text(term) in doc_norm
            )
            # If the contract has a long, specific term ("距离函数",
            # "微分形式"), one hit is enough to make the lecture worth
            # scanning. Otherwise require two groups to avoid generic words.
            strong = any(len(_match_text(term)) >= 3 for term in specific)
            required = 1 if strong else max(1, min(2, len(groups)))
            if specific_hits < required:
                if not any(term.casefold() in doc.text.casefold()
                           for term in bridge_terms):
                    continue
        legacy = legacy_candidate_windows(
            doc.turns, spec, max_candidates=6
        )
        for candidate in legacy:
            candidate.source = doc.slug
            candidate.source_date = doc.date
            candidate.source_part = doc.part
            candidate.origin = "legacy"
        if legacy:
            legacy_by_doc[doc.slug] = legacy
            doc_scores[doc.slug] = max(
                c.coverage * 20.0 + min(c.keyword_score, 60.0) for c in legacy
            )
        doc_new: list[Candidate] = []
        for candidate in candidate_windows(
            doc.turns, spec, max_candidates=max(1, per_doc),
            bridge_terms=bridge_terms,
        ):
            if candidate.coverage < min_coverage:
                continue
            candidate.source = doc.slug
            candidate.source_date = doc.date
            candidate.source_part = doc.part
            candidate.origin = "new"
            doc_new.append(candidate)
            pool.append(candidate)
        new_by_doc[doc.slug] = doc_new
    pool.sort(
        key=lambda c: (-(c.coverage * 100.0 + min(c.keyword_score, 60.0)), -c.coverage)
    )
    # Full union: keep the top legacy and top new candidates from every
    # eligible lecture. The host sees a bounded, short-excerpt view; no
    # lecture is dropped just because another lecture scored higher.
    def cand_score(c: Candidate) -> float:
        return c.coverage * 100.0 + min(c.keyword_score, 60.0)

    selected: list[Candidate] = []

    def add_candidate(candidate: Candidate) -> None:
        for kept in selected:
            if kept.source == candidate.source and _overlap_ratio(candidate, kept) >= 0.6:
                return
        selected.append(candidate)

    for doc in docs:
        for candidate in legacy_by_doc.get(doc.slug, [])[:6]:
            if candidate.coverage >= min_coverage:
                add_candidate(candidate)
        for candidate in new_by_doc.get(doc.slug, [])[:per_doc]:
            if candidate.coverage >= min_coverage:
                add_candidate(candidate)
    selected.sort(key=lambda c: (-cand_score(c), c.source, c.start))
    return selected[:max_candidates]
    return selected


def prepare_host_request_corpus(
    docs: list[TranscriptDoc],
    spec: dict,
    max_candidates: int = 300,
    per_doc: int = 12,
    min_coverage: float = 0.4,
    bridge_terms: list[str] | None = None,
) -> dict:
    """Host request over a whole course folder; each candidate carries its source."""
    candidates = candidate_windows_corpus(
        docs, spec, max_candidates=max_candidates, per_doc=per_doc,
        min_coverage=min_coverage, bridge_terms=bridge_terms,
    )
    by_slug = {doc.slug: doc for doc in docs}
    items = []
    for rank, candidate in enumerate(candidates, 1):
        doc = by_slug.get(candidate.source)
        if doc is None:
            continue
        _selected, excerpt = build_excerpt(doc.turns, candidate.start, candidate.end)
        if len(excerpt) > 1000:
            excerpt = excerpt[:600] + "\n...\n" + excerpt[-350:]
        items.append({
            "candidate_id": rank,
            "source": candidate.source,
            "date": candidate.source_date,
            "part": candidate.source_part,
            "start": round(candidate.start, 2),
            "end": round(candidate.end, 2),
            "start_text": fmt_ts(candidate.start),
            "end_text": fmt_ts(candidate.end),
            "coverage": round(candidate.coverage, 3),
            "keyword_score": round(candidate.keyword_score, 3),
            "excerpt": excerpt,
        })
    return {
        "knowledge_point": {
            "id": spec.get("id", ""),
            "name": spec.get("name", ""),
            "definition": spec.get("definition", ""),
            "must_include": spec.get("must_include", []),
            "mention_only": spec.get("mention_only", ""),
            "bridge_terms": list(bridge_terms or spec.get("bridge_terms", [])),
        },
        "instructions": HOST_INSTRUCTIONS,
        "candidates": items,
        "sources": [
            {"slug": doc.slug, "date": doc.date, "part": doc.part}
            for doc in docs
        ],
    }


def apply_host_response_corpus(
    docs: list[TranscriptDoc],
    spec: dict,
    request: dict,
    response: dict,
    core_min: float = 30.0,
    core_max: float = 90.0,
) -> dict:
    """Apply host judgments when candidates come from different lectures."""
    by_slug = {doc.slug: doc for doc in docs}
    candidates = {
        int(item["candidate_id"]): item for item in request.get("candidates", [])
    }
    judged = []
    teach_by_id: dict[int, dict] = {}
    mention_by_id: dict[int, dict] = {}

    for judgment in response.get("judgments") or []:
        try:
            candidate_id = int(judgment.get("candidate_id"))
        except (TypeError, ValueError):
            continue
        item = candidates.get(candidate_id)
        if not item:
            continue
        doc = by_slug.get(str(item.get("source") or ""))
        if doc is None:
            continue
        level = str(judgment.get("level") or "unrelated").strip().lower()
        if level not in {"teach", "mention", "unrelated"}:
            level = "unrelated"
        verify = judgment.get("verify")
        if level == "teach" and verify is False:
            level = "mention"
        confidence = float(judgment.get("confidence") or 0.0)
        judged.append({
            "rank": candidate_id,
            "source": item.get("source"),
            "date": item.get("date"),
            "part": item.get("part"),
            "start": item.get("start"),
            "end": item.get("end"),
            "start_text": item.get("start_text"),
            "end_text": item.get("end_text"),
            "coverage": item.get("coverage"),
            "level": level,
            "teach": level == "teach",
            "mention": level == "mention",
            "confidence": confidence,
            "verify": judgment,
            "reason": str(judgment.get("reason") or ""),
        })
        if level not in {"teach", "mention"}:
            continue
        selected, _excerpt = build_excerpt(
            doc.turns, float(item["start"]), float(item["end"]))
        if not selected:
            continue
        payload = {
            "level": level,
            "confidence": confidence,
            "reason": str(judgment.get("reason") or ""),
            "start": judgment.get("start"),
            "end": judgment.get("end"),
            "_verify": judgment,
            "_evidence_verified": judgment.get("evidence_verified"),
        }
        segment = _build_segment(
            doc.turns, selected, payload, spec,
            Candidate(
                float(item["start"]), float(item["end"]),
                selected[0][0], selected[-1][0],
                source=doc.slug, source_date=doc.date, source_part=doc.part,
            ),
            candidate_id, len(candidates),
            "teach" if level == "teach" else "mention_only",
            core_min, core_max,
        )
        segment["source"] = {
            "slug": doc.slug, "date": doc.date, "part": doc.part,
            "path": str(doc.path),
        }
        (teach_by_id if level == "teach" else mention_by_id)[candidate_id] = segment

    def choose(by_id: dict[int, dict]) -> dict:
        segments = list(by_id.values())
        selected_ids = _response_selected_ids(response)
        chosen = [by_id[item] for item in selected_ids if item in by_id]
        if not chosen:
            return _select_segment(segments, response)
        if len(chosen) == 1:
            return chosen[0]
        # Multiple selected candidates: merge when they come from one lecture.
        sources = {item["source"]["slug"] for item in chosen}
        if len(sources) == 1:
            doc = by_slug[sources.pop()]
            start = min(float(item["start"]) for item in chosen)
            end = max(float(item["end"]) for item in chosen)
            # Do NOT use build_excerpt here: it caps to 180s and would cut a
            # merged "continuity + differentiability" pair in half.
            selected = [
                (index, turn) for index, turn in enumerate(doc.turns)
                if turn.end > start and turn.start < end
            ]
            if selected:
                payload = {
                    "level": "teach",
                    "confidence": max(
                        item["semantic"]["confidence"] for item in chosen
                    ),
                    "reason": str(response.get("selection_reason") or ""),
                    "start": 0,
                    "end": len(selected) - 1,
                }
                merged = _build_segment(
                    doc.turns, selected, payload, spec,
                    Candidate(start, end, selected[0][0], selected[-1][0]),
                    chosen[0]["semantic"]["candidate_rank"], len(candidates),
                    "teach", core_min, core_max,
                )
                merged["source"] = {
                    "slug": doc.slug, "date": doc.date, "part": doc.part,
                    "path": str(doc.path),
                }
                merged["semantic"]["merged_candidate_ids"] = [
                    item["semantic"]["candidate_rank"] for item in chosen
                ]
                return merged
        return _select_segment(chosen, response)

    if teach_by_id:
        best = choose(teach_by_id)
        return {"segments": [best], "decision": "teach",
                "reason": best["semantic"]["reason"], "candidates": judged,
                "selection_reason": str(response.get("selection_reason") or "")}
    if mention_by_id:
        best = choose(mention_by_id)
        best["semantic"]["reason"] = (
            "只找到相关提及/未完整展开，未确认是在完整讲解："
            + str(best["semantic"].get("reason") or "")
        )
        return {"segments": [best], "decision": "mention_only",
                "reason": best["semantic"]["reason"], "candidates": judged,
                "selection_reason": str(response.get("selection_reason") or "")}
    return {
        "segments": [],
        "decision": "not_found",
        "reason": "宿主判定所有候选均为提及/无关，未找到教学段。",
        "candidates": judged,
    }


def spec_from_query(query: str, keywords: list[str] | None = None) -> dict:
    """Minimal spec for CLI compatibility when the user did not supply one."""
    must = list(keywords or []) or [query]
    return {
        "name": query,
        "definition": query + "；需要讲清定义、条件、结论或方法，而不是只出现术语。",
        "must_include": must,
        "mention_only": "只在考试范围、复习清单、或顺带一提不算。",
    }


def load_specs(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("specs") or data.get("knowledge_points") or []
    return {str(item["id"]): item for item in data}
