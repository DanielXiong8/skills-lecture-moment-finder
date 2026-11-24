# 语义命中规则（host 模式）

关键词匹配只能判断“这个词有没有出现”，不能判断“这段是不是在教这个知识点”。
语义模式把判定交给**宿主 agent 自己的模型**（也就是调用这个 Skill 的模型），
Skill 脚本只负责确定性的召回、提示词和窗口应用。

```text
脚本：关键词/必含项召回 Top-K 候选
  → 生成 judge_request.json（候选原文 + 知识点契约 + 判定规则）
宿主 agent：用自己的模型判断 teach / mention / unrelated，并给出起止句
  → 写 judge_response.json
脚本：按判定生成教学窗口；全部 unrelated 时返回 not_found
```

这样不需要在 Skill 里打包本地大模型，判定质量也由宿主模型决定。

## 第 1 步：生成判定请求

```bash
python scripts/find_moment.py \
  --transcript path/to/lecture.json \
  --query "第二型曲面积分" \
  --specs kp_specs.json \
  --knowledge-point kp42 \
  --semantic host \
  --judge-request-out out/kp42_request.json
```

## 第 2 步：宿主 agent 判定

Agent 读取 `out/kp42_request.json`，按里面的 `instructions` 对每个 candidate 判定，
写出 `out/kp42_response.json`：

```json
{
  "best_candidate_id": 2,
  "selected_candidate_ids": [2],
  "selection_reason": "候选2是二重积分极坐标换元；候选1是三重积分",
  "judgments": [
    {"candidate_id": 2, "level": "teach", "start": 0, "end": 40,
     "verify": true, "confidence": 0.95, "reason": "明确用定向曲面/法向量定义通量积分"}
  ]
}
```

`best_candidate_id` / `selected_candidate_ids` 由**宿主模型**选，不由脚本按顺序或
confidence 猜。限定词必须匹配：二重积分不能被三重积分顶替，可导性不能被连续性顶替。
一个候选只覆盖定义的一部分时，可以选多个同来源、时间相邻的候选，脚本会合并。

判定标准：

- `teach`：定义、解释、证明、推导或完整例题说明该知识点；
- `mention`：相关提及、方法列表、复习/考试范围、只讲到一半、或在别的证明里顺带用到；
- `unrelated`：无关或关键词巧合；
- `verify`：`teach` 时是否明确陈述了 `definition` 里的具体命题/条件/公式；只给结论名称填 `false`。

## 第 3 步：应用判定，生成结果

```bash
python scripts/find_moment.py \
  --transcript path/to/lecture.json \
  --query "第二型曲面积分" \
  --specs kp_specs.json \
  --knowledge-point kp42 \
  --semantic host \
  --judge-response out/kp42_response.json \
  --out out/kp42
```

### 跨讲检索

一个知识点可能不是在“主讲次”讲透的。给一个目录，每个文件一讲：

```bash
python scripts/find_moment.py \
  --transcripts-dir lectures/ \
  --query "反函数定理" \
  --specs kp_specs.json --knowledge-point kp23 \
  --semantic host --judge-request-out out/kp23_request.json
```

请求里的每个 candidate 都带 `source`（讲次）、`date`、`part`；宿主判定后，
`result.json` 的 segment 也带 `source`。脚本不再先选一个主讲次，避免“真正的
讲解在另一讲、候选池里根本没有”的结构性漏召。

### 没教的知识点：LLM 推断后二次检索

如果第一轮全是 `unrelated` / `mention`，让宿主模型读这段 transcript 和题目，
写出老师实际用过的等价说法或前置知识点，再检索：

```bash
python scripts/find_moment.py ... \
  --bridge-terms "截面法,积分求体积,旋转体体积" \
  --semantic host --judge-request-out out/retry_request.json
```

`bridge_terms` 会参与候选召回，但不改变原始 `must_include` 的覆盖率口径；它用于
“课程没按知识点名讲，但用了另一个说法”的情况。若仍然没有 `teach`，就按
`not_found` 返回，不要硬凑。

## 知识点契约

每个知识点需要一份“契约”（定义、必须讲到、只算提及）：

```json
{
  "id": "kp42",
  "name": "第二型曲面积分",
  "definition": "把定向曲面 S 上向量场 F=(P,Q,R) 的通量定义为第二型曲面积分 ∬ P dy∧dz + Q dz∧dx + R dx∧dy；要说明定向/法向量、与第一型曲面积分的关系。",
  "must_include": ["第二型曲面积分|第二型", "定向|法向量", "通量|flux", "P|Q|R"],
  "mention_only": "只在考试范围列出名称、没给定义不算。"
}
```

## 可选：本地 Qwen fallback

没有宿主 agent、需要纯离线 CLI 时，才用本地 Qwen：

```bash
python scripts/find_moment.py ... --semantic llm
```

需要 `llama-cpp-python` 和本地 GGUF 模型（用 `--model` 或
`LECTURE_SEMANTIC_MODEL` 指定，不写死路径）。一个知识点约 30–50 秒。
本地模型不可用时回退关键词匹配；`--semantic llm` 会显式报错。

## 课程没讲到：AI 补充讲解

如果两轮检索都是 `not_found`，或只返回 `mention_only`（没讲透），不要让模型硬凑课堂片段，也不要把 AI 内容伪装成课堂原话。
加 `--ai-fallback` 生成一份补充讲解请求：

```bash
python scripts/find_moment.py ... --ai-fallback --out out/kp37
# → out/kp37/ai_fallback_request.json
```

宿主/LLM 按请求里的 `response_format` 写回：

```json
{
  "title": "锥体体积与截面法",
  "explanation": "用平行于底面的截面面积 A(y)=A(1-y/h)^2 积分，得到 V=Sh/3。",
  "key_points": ["截面面积随高度二次变化", "积分求体积"],
  "example": "底面 S、高 h 的锥体",
  "caveat": "课程录播中未找到该知识点的完整讲解，以上为 AI 补充。"
}
```

再应用：

```bash
python scripts/find_moment.py ... --ai-fallback   --ai-fallback-response out/kp37/ai_fallback_response.json --out out/kp37
```

输出会带 `ai_fallback.source = "ai"` 和固定 warning：
“课程录播中未找到该知识点的完整讲解；以下为 AI 补充，不是课堂原话，也没有时间戳。”
`result.json` 不含课堂时间戳，不计入严格召回；它是补课答案，不是证据。

## 发布版评测口径（单讲 v1）

发布版默认使用**单讲、6 候选**的 v1 证据路径，保证代码行为与评测数字一致：

| 指标 | 结果 |
|---|---:|
| 评测集 | 34 个录播 / 55 个知识点 / 69 道题 |
| 严格召回 | 58.18% |
| 可用召回 | 90.91% |
| 按题严格成功 | 57.97% |
| 按题可用成功 | 97.10% |

- 严格：完整讲解，不含提及；
- 可用：完整讲解 + 相关提及；
- 课堂片段和 AI 补讲分开统计；AI 补讲不进入严格/可用召回。

跨讲检索、候选并集和 LLM 选候选是实验路径（`--transcripts-dir`），
其 55 点并集 benchmark 为严格 74.55% / 可用 98.18%，
但**不是发布版默认路径的实测数字**，不要混用。

## 输出字段

`result.json` 会增加：

```json
{
  "semantic": {
    "decision": "teach | mention_only | not_found",
    "reason": "为什么判为讲解/提及/未找到",
    "selection_reason": "为什么选这个候选（LLM 给的限定词匹配理由）",
    "candidates": [
      {"candidate_id": 2, "source": "2026-01-10",
       "start_text": "11:40", "level": "teach", "confidence": 0.9,
       "verify": true, "reason": "..."}
    ]
  }
}
```

跨讲模式下每个 `segments[]` 还带：

```json
{"source": {"slug": "lecture-17", "date": "2026-01-17",
            "part": "", "path": "lectures/lecture-17.json"}}
```
