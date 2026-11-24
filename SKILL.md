---
name: lecture-moment-finder
description: |
  课堂实时定位 + 课后回放。上课前写下你关心的问题，启动后自动录音/录屏、
  实时转写；命中后持续跟踪，直到老师讲完这个知识点，自动保存完整区间
  的音频、屏幕录像和原话（例如 01:30–06:30 全程）；课后也可以对整段转写
  做“老师讲 XX 的那几分钟”的定位。默认本机运行；只有显式 `--semantic openai` 才把转写窗口发到你指定的端点，不上传音视频。
  Triggers include: "开始上课/开始记录", "老师讲……的那几分钟", "实时定位课堂重点",
  "录音里找老师讲……", "课堂回放定位", "lecture moment finder",
  "start a live lecture capture", "find the minute my teacher covered",
  以及用户给出课堂录音/转写并询问“讲到哪一段/哪几分钟/哪里提到”时。
tags: [education, lecture, live-capture, find-moment, replay, offline, learning-skill]
---

# 课堂实时定位 + 回放（Lecture Moment Finder）

## Overview

两种用法，共用一个定位内核：

```text
实时模式（上课中）
  录音/录屏 + 课前关心清单 → 实时转写 → LLM 语义判定 → 命中 → 跟踪到主题结束
  → 完整区间的音频 + 屏幕录像 + 原话

课后模式（下课后）
  完整转写 + 一句问题 → 自适应完整区间 + 核心 30–90 秒 → 结果与回放材料
```

它不做会议纪要，也不生成课程总结。它只做一件事：**在你关心的地方留下可回放的完整片段**。

适用场景：

- 上课：老师讲到“考试范围 / 某个例题 / 某个公式”时，自动保存那一段；
- 补课：网课太长，先跳到没听懂的那一段；
- 教研/培训：从多段录课里找到某个知识点首次出现的位置；
- 内容二创：从长录屏里定位可以剪成短视频的片段。

## Prerequisites

- **Python 3.10–3.12**：实时模式需要 3.10+；3.13 移除了 `audioop`，不支持。
- **课后模式**：转写定位是纯标准库；如果要生成 `clip_core.mp4` / `clip_full.mp4` /
  关键帧，需要 **ffmpeg 或 PyAV（二选一）**。
- **实时模式**：首次运行装依赖：
  `python scripts/ensure_deps.py --model base [--mirror]`，
  会安装 `pyaudiowpatch`、`mss`、`Pillow`、`numpy`、`faster-whisper`、`av`
  （`av`/PyAV 负责屏幕录像和课后切片，不需要外部 ffmpeg）。
- **Windows**：系统声音环回依赖 WASAPI；线下课可改用麦克风。
- **依赖检查（统一入口）**：`python scripts/doctor.py [--json]`。
  它一次报告 ffmpeg、PyAV、faster-whisper、音频/屏幕依赖和本地模型，
  并给出视频切片引擎 / 实时模式是否就绪。
- **依赖解析顺序（所有脚本共用 `deps.py`）**：
  1. 显式 `--ffmpeg` / 环境变量（`LECTURE_FFMPEG`、`FFMPEG_BINARY`）
  2. 本机缓存（`~/.lecture-moment-finder/deps_cache.json`）
  3. `PATH`
  4. 常见目录有界搜索（用户目录、Desktop/Downloads/Documents、
     Program Files 等；有深度和时间上限；其他盘用 `--root` 显式指定）
  5. fallback：视频切片用 PyAV；语义判定用 host 模型
- 有 ffmpeg 时优先用；没有 ffmpeg 时自动用 PyAV，**不会再出现“没有 ffmpeg 所以没有视频”**。
  课后定位首轮候选无 teach 时，宿主可回 `retry_terms` 触发第二轮检索（`--retry-terms`），仍无 teach 才 AI 补讲。

## Workflow

1. **课前**：复制 `examples/interests.example.json`，写下关心的问题（`definition` 可选，`keywords` 只作提示）。
2. **启动**：`live_ctl.py doctor` → `devices` → `start --consent`。
3. **上课**：你（CodeArts）每 20–30 秒读 `judge/pending.json`，用自己的模型判定；`watch` 看命中并继续跟踪。
4. **下课**：`stop`；在 `hits/hit_NNN/` 拿到完整区间音频、屏幕录像和原话。
5. **复习**：直接回放命中的 `audio.wav`，或用 `find_moment.py` 对完整转写做二次定位。

## Core Commands

```bash
python scripts/doctor.py                          # 统一依赖报告（ffmpeg/PyAV/模块/模型）
python scripts/live_ctl.py doctor                 # 依赖 + 设备检查
python scripts/live_ctl.py devices                # 列出音频设备/显示器
python scripts/live_ctl.py start --config ... --out ... --consent
python scripts/judge_live.py next --out ...       # 可选：读 pending 并打印请求
python scripts/judge_live.py submit --out ... --request req_0001 --file resp.json
python scripts/live_ctl.py var --out ...          # 手动 VAR：前后各 60 秒
python scripts/live_ctl.py watch --out ...        # 看命中（不停止引擎）
python scripts/live_ctl.py stop --out ...         # 优雅停止
python scripts/live_ctl.py simulate --config ... --transcript ... --out ... --speed 0
python scripts/find_moment.py --transcript ... --query "老师讲 XX 的那几分钟"
python scripts/selftest.py                        # 离线 + 实时完整区间自测
```

## Live Mode（上课中，主模式）

### 它怎么工作

```text
课前：interests.json（你关心的问题 + 可选定义）
  ↓
启动：录音（系统声音/麦克风）+ 录屏（可选），本地 faster-whisper 转写
  ↓
语义：每 20–30 秒把最近 45 秒转写交给 LLM，判 teach / mention / none
      host   = CodeArts agent 自己的模型（本机文件请求/响应）
      openai = 任意 OpenAI 兼容端点
  ↓
命中：只有 teach 才开区间；mention / none 不命中；不按关键词触发
  ↓
跟踪：后续窗口继续 teach 就延长；连续两个窗口不再教 → 收尾
  ↓
保存：完整区间的音频 + 屏幕录像 + 转写 + 原话 → hits/hit_NNN/
  ↓
课后：可继续用完整转写做二次定位
```

> 实时触发**没有关键词回退**。LLM 后端不可用时，引擎只保留手动 VAR，
> 不会退化成关键词匹配——关键词误命中正是这个模式要解决的问题。

### 课前：写关心清单

复制 `examples/interests.example.json`，改 `interests`：

```json
{
  "session_id": "lecture-2026-01-10",
  "title": "数据结构 第 7 讲",
  "consent": false,
  "interests": [{
    "id": "dp",
    "question": "老师讲动态规划状态转移方程的那几分钟",
    "definition": "写出状态定义、转移方程、边界和计算顺序，并用例题说明",
    "keywords": ["动态规划", "状态转移方程"]
  }],
  "semantic": {
    "enabled": true, "backend": "host",
    "window_seconds": 45.0, "interval_seconds": 25.0,
    "min_new_chars": 80, "min_confidence": 0.6, "max_pending": 2
  },
  "trigger": {
    "ring_seconds": 300.0, "min_span": 45.0, "max_span": 600.0,
    "semantic_grace_seconds": 90.0, "manual_span_seconds": 120.0
  }
}
```

`semantic` 关键字段：`enabled`（false=只手动 VAR）、`backend`（`host` 或
`openai`）、`window_seconds`（每次回看）、`interval_seconds`（判定间隔）、
`min_confidence`（低于该值的 `teach` 只算 `mention`）。`trigger` 关键字段：
`ring_seconds`（可回溯缓冲）、`min_span`/`max_span`（区间上下限）、
`semantic_grace_seconds`（多久没有新 `teach` 就收尾）、`manual_span_seconds`
（手动 VAR 窗口）。完整字段和默认值见 [live-mode.md](references/live-mode.md)。

`keywords` 只作为 LLM 的提示，不参与触发；`capture.source` 可选
`loopback`（系统声音，推荐）、`mic`（麦克风）、`both`（混音）。

### 启动前：检查环境和设备

```bash
# 依赖是否齐全（pyaudiowpatch / mss / Pillow / numpy / faster-whisper）
python scripts/live_ctl.py doctor

# 列出音频设备和显示器
python scripts/live_ctl.py devices
```

如果 `doctor` 显示缺依赖，可以在另一个已经装好依赖的 Python 环境里运行：

```bash
"<path-to-python>" scripts/live_ctl.py doctor
```

### 启动 / 查看 / 停止

```bash
# 启动（必须确认已获得录音/录屏许可）
python scripts/live_ctl.py start \
  --config examples/interests.example.json \
  --out out/lecture-2026-01-10 \
  --consent

# 查看状态
python scripts/live_ctl.py status --out out/lecture-2026-01-10

# 持续看新命中（Ctrl+C 只退出监听，不停止引擎）
python scripts/live_ctl.py watch --out out/lecture-2026-01-10

# 优雅停止
python scripts/live_ctl.py stop --out out/lecture-2026-01-10
```

> 当前 Python 没有采集依赖时，用 `--python "<path-to-python>"` 指定另一个解释器。
> Skill 不写死任何本机路径。

### CodeArts 怎么判（默认 host）

**你（CodeArts）就是判定模型**，不用外部 API，也不用关键词。引擎已经把最近
45 秒转写和兴趣定义写成文件：

1. 读 `out/.../judge/pending.json`；有 pending 就打开 `path` 指向的请求。
2. 用你自己的模型对 `request.interests` 逐条判 `teach / mention / none`。
   需要更长上下文时，直接读同目录的 `transcript.jsonl`；判定仍以 `request.window` 为准。
3. 把结果写到 `out/.../judge/responses/<request_id>.json`：
   `{"judgments":[{"interest_id":"dp","level":"teach","confidence":0.92,`
   `"start":12.0,"end":38.0,"quote":"原话","reason":"正在写转移方程"}]}`
4. 每 20–30 秒重复；没有 pending 就等下一轮，**不要用关键词代替**。

判定标准：`teach`=正在定义/解释/推导/证明/完整例题；`mention`=只提名字、
考试范围或顺带一句；`none`=没讲。**只出现关键词不算 teach。**
`judge_live.py next/submit` 只是可选的读写助手；你直接用 Read/Write 工具也一样。
如果当前回合不能常驻，就在每次获得控制权时先补处理 pending，并提醒用户保持会话。

### 手动 VAR

任何时刻都可以手动标记当前片段，不依赖 LLM：

```bash
python scripts/live_ctl.py var --out out/lecture-2026-01-10
# 可选：--at 320.5 指定会话内秒数；--note "这里没听懂"
```

引擎抓取该时刻前后各 60 秒，写入 `hits/hit_NNN/`，`hit.json` 里
`trigger="manual"`、`close_reason="manual_end"`。

### 使用 OpenAI 兼容端点

不想跑宿主循环时，可以指向任意 OpenAI 兼容服务（包括本机网关）：

```bash
export LECTURE_LLM_BASE_URL="http://127.0.0.1:PORT/v1"
export LECTURE_LLM_MODEL="your-model"
export LECTURE_LLM_API_KEY="..."   # 可选，按端点要求
python scripts/live_ctl.py start ... --semantic openai
```

也可以写在配置里：`semantic.backend="openai"`，
`semantic.openai.base_url/model/api_key_env`。端点不可用时引擎报错并停止自动命中，
不会回退关键词。**openai 模式会把转写窗口发到该端点；host 模式只在本机读写文件。**

### 实时模式输出

```text
out/lecture-2026-01-10/
├── status.json          # 当前状态、最近一句、已完成/正在抓取的命中
├── transcript.jsonl     # 实时转写（ts/end/text）
├── hits.jsonl           # 命中事件流（hit_open / hit_close）
├── session.json         # 课后汇总：配置、耗时、文件大小
├── engine.log           # 后台引擎日志
└── hits/
    └── hit_001/
        ├── audio.wav    # 完整知识点区间（例如 01:30–06:30）
        ├── screen.mp4   # 同一区间的屏幕录像（5fps / 1280 宽）
        ├── transcript.txt
        └── hit.json     # 问题、原话、完整区间、核心片段、结束原因
```

> 屏幕录像用 PyAV 在本机编码，不依赖外部 ffmpeg；如果 PyAV 不可用，会自动回退为
> `screens/` 关键帧截图。学习 Skill 只保存**可回放材料**，不生成
> `manifest.json` / `SHA256SUMS.txt` / zip；需要文件完整性哈希时，用法律取证 Skill
> （`recording-evidence-finder`）。

### 没有设备时：回放转写

用一段转写模拟“上课过程”，不打开麦克风/屏幕，适合演示和 QA：

```bash
python scripts/live_ctl.py simulate \
  --config examples/interests.example.json \
  --transcript examples/lecture-transcript.json \
  --out out/simulated-lecture \
  --speed 10
```

`--speed 0` 表示立即跑完；`--speed 10` 表示 10 倍速。

### 调参

| 现象 | 调整 |
|---|---|
| 命中太频繁 | 提高 `semantic.min_confidence`、缩小 `window_seconds`、把 `definition` 写具体 |
| 总是漏掉 | 降低 `min_confidence`、增大 `window_seconds`、补充 `definition` |
| 判定太慢/积压 | 增大 `interval_seconds`、减小 `max_interests`；host 改 openai 降延迟 |
| 一个知识点被拆成多段 | 提高 `semantic_grace_seconds`、`max_span` |
| 区间太长 | 降低 `semantic_grace_seconds`、`max_span` |
| 区间太短/提前收尾 | 提高 `semantic_grace_seconds`、`max_gap` |
| 转写跟不上 | 用 `model: "tiny"`，或增大 `step_seconds` |
| 转写太碎 | 增大 `chunk_seconds`，或换 `model: "small"` |
| 录屏文件太大 | 降低 `video_fps` / `video_width`，或提高 `video_crf` |
| 不想存屏幕 | `capture.screen: false` |

### 实时模式的边界

- **Windows 优先**：系统声音环回依赖 WASAPI；macOS/Linux 需要换采集实现。
- **本地模型**：默认 `base`，CPU 可跑；模型越大越准也越慢。
- **转写不是逐字稿**：分块 ASR 会在句首/句尾有少量误差，命中片段以原话+音频为准。
- **不做说话人分离**：在线课通常只有老师一路声音；需要区分说话人时再扩展。
- **语义触发，不靠关键词**：只有 LLM 判 `teach` 才命中；没有 LLM 后端时只保留手动 VAR。
- **host 模式延迟**：判定由宿主 agent 完成，命中会晚一个判定周期；需要低延迟时用 `openai` 端点。
- **屏幕录像**：默认 5fps、1280 宽，用于回放定位；没有 PyAV 时自动回退为关键帧截图。

## Post-class Mode（下课后，回放定位）

### 定位并生成回放材料

```bash
python scripts/find_moment.py \
  --transcript examples/lecture-transcript.json \
  --query "老师讲动态规划状态转移方程的那几分钟" \
  --out out/dp-moment
```

有原始录屏时，加上 `--media`，会额外生成核心片段、完整讨论片段和关键帧：

```bash
python scripts/find_moment.py \
  --transcript path/to/lecture.srt \
  --query "老师讲考试范围的那几分钟" \
  --media path/to/lecture.mp4 \
  --out out/exam-scope
```

只查看结果、不写文件：

```bash
python scripts/find_moment.py \
  --transcript path/to/lecture.srt \
  --query "老师讲二叉树遍历" \
  --no-save
```

### 语义命中规则（推荐：宿主模型判定）

关键词只能判断“词有没有出现”，不能判断“这段是不是在教”。语义模式由宿主 agent
用自己的模型判 `teach / mention / unrelated`，脚本只做召回、生成请求和应用窗口；
全部 unrelated 时返回 `not_found`。

```bash
# 1) 单讲生成请求
python scripts/find_moment.py ... --semantic host --judge-request-out out/req.json
# 跨讲：每个文件一讲，候选带 source，不再只搜一个主讲次
python scripts/find_moment.py --transcripts-dir lectures/ ... \
  --semantic host --judge-request-out out/req.json
# 2) 宿主写 resp.json：逐条 level/start/end/verify，并给 best_candidate_id /
#    selected_candidate_ids，由你选最贴合“二重/三重、连续/可导”等限定词的候选
# 3) 应用判定
python scripts/find_moment.py ... --semantic host --judge-response out/resp.json --out out/result
```

第一轮 `not_found` 时：读 transcript 和题目，推断老师实际用了什么说法/前置知识点，
再用 `--bridge-terms "截面法,积分求体积"` 重跑；跨讲时加 `--transcripts-dir`。仍是 `not_found` 或只有 `mention` 就加 `--ai-fallback`：生成 AI 补充讲解，明确标注“不是课堂原话、没有时间戳”。

详细契约、判定标准、输出字段见 [semantic-mode.md](references/semantic-mode.md)。
本地 Qwen 只做 `--semantic llm` 离线 fallback，不写死模型路径。

### 窗口机制

- 相邻句仍然在讲同一主题 → 继续扩展；
- 遇到长静音（默认 20 秒）、课间休息、相关度明显下降 → 停止；
- 上限按录音长度自动计算（默认 30%，最多 10 分钟），避免扩到整场课。

完整区间内部再挑**最密的 30–90 秒**作为核心片段。这样既能看到“老师从哪讲到哪”，
又能一键回放最关键的那几十秒。

### 输出文件

```text
out/dp-moment/
├── result.json             # 完整区间 + 核心片段 + 命中理由
├── report.md               # 可读报告
├── transcript_excerpt.txt  # 命中区间的原话
├── clip_core.mp4           # 有 --media 时生成（ffmpeg 或 PyAV）
├── clip_full.mp4           # 同上；完整讨论 ≤10 分钟时生成
└── frames/                 # 核心片段的关键帧
```

不生成 `manifest.json` / `SHA256SUMS.txt` / zip；学习 Skill 只负责定位和回放。

### 可选：本地生成转写

```bash
python scripts/transcribe.py path/to/lecture.mp4 --out lecture.json --model small
```

需要 `faster-whisper`；视频解码用 PyAV，通常不需要外部 ffmpeg。CPU 可跑；模型越大越准也越慢。

## Parameter Confirmation

### 实时模式

| 参数/字段 | 必填 | 说明 |
|---|---|---|
| `--config` | 是 | `interests.json` 路径 |
| `--out` | 是 | 会话输出目录 |
| `--consent` | 是 | 确认已获得录音/录屏许可 |
| `--python` | 否 | 有采集依赖的 Python；缺省用当前解释器 |
| `interests[].question` | 是 | 一句具体问题，语义判定的主依据 |
| `interests[].definition` | 否 | “讲到什么才算教”；最能减少误判 |
| `interests[].keywords` | 否 | 只作 LLM 提示，不参与触发 |
| `semantic.enabled` | 否 | 默认 true；false=只手动 VAR |
| `semantic.backend` | 否 | `host`（默认）/ `openai` |
| `semantic.min_confidence` | 否 | 低于该值的 `teach` 只算 `mention`，默认 0.6 |
| `--semantic` | 否 | `auto` / `off` / `host` / `openai`；不会回退关键词 |
| `capture.source` | 否 | `loopback` / `mic` / `both` |
| `capture.screen` | 否 | 是否录屏 |
| `capture.video_fps` | 否 | 录屏帧率，默认 5 |
| `capture.video_width` | 否 | 录屏宽度，默认 1280 |
| `capture.video_crf` | 否 | 录屏质量（CRF 越大文件越小），默认 28 |
| `capture.monitor` | 否 | 录制哪个显示器，默认 1（主屏） |
| `asr.model` | 否 | `tiny` / `base` / `small` / `none` |
| `trigger.ring_seconds` | 否 | 音频/屏幕缓冲秒数，默认 300 |
| `trigger.max_span` | 否 | 单个区间上限，默认 600 秒 |
| `trigger.semantic_grace_seconds` | 否 | 没有新 `teach` 多久收尾，默认 90 秒 |
| `trigger.manual_span_seconds` | 否 | 手动 VAR 窗口，默认 120 秒 |

### 课后模式

| 参数 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `--transcript` | 二选一 | 带时间戳的转写文件 | `lecture.srt` |
| `--transcripts-dir` | 二选一 | 跨讲检索；目录里每个文件一讲 | `lectures/` |
| `--bridge-terms` | 否 | LLM 推断的等价说法/前置知识点，二次检索 | `截面法,积分求体积` |
| `--query` | 是 | 一个具体知识点 | `老师讲状态转移方程` |
| `--out` | 否 | 输出目录，默认 `./out` | `out/dp` |
| `--media` | 否 | 原始音视频；有才会生成 clip | `lecture.mp4` |
| `--ffmpeg` | 否 | ffmpeg.exe 路径；PATH 里没有时用它（也可用 `LECTURE_FFMPEG`） | `<path-to-ffmpeg.exe>` |
| `--no-deps-search` | 否 | 不搜索常见目录，只查显式/env/缓存/PATH + PyAV fallback | — |
| `--min-span` / `--max-span` | 否 | 完整讨论区间；`max-span=0` 自动 | `45` / `0` |
| `--core-min` / `--core-max` | 否 | 核心片段秒数，默认 30–90 | `30` / `90` |
| `--semantic` | 否 | `auto`（默认，有 host 请求/响应就用 host，否则关键词）/ `off` / `host` / `llm` | `host` |
| `--spec` / `--specs` | 否 | 单个/多个知识点契约 JSON | `kp_specs.json` |
| `--knowledge-point` | 否 | 从 `--specs` 里选哪个 id | `kp42` |
| `--model` | 否 | 本地 GGUF 模型路径（不写死） | `<model>.gguf` |
| `--max-candidates` | 否 | 最多语义判定几个候选，默认 5 | `6` |
| `--min-coverage` | 否 | 候选必须覆盖多少必含项，默认 0.5 | `0.5` |
| `--min-confidence` | 否 | 第一轮“是否在讲”阈值，默认 0.6 | `0.6` |
| `--min-verify-confidence` | 否 | 第二轮“具体命题确认”阈值，默认 0.6 | `0.6` |
| `--no-save` | 否 | 只打印结果，不写文件 | — |

如果用户没有转写、也暂时不想装 `faster-whisper`，直接告诉他：
“先导出字幕（SRT/VTT）也可以，效果一样。”

## Privacy and Limits

- **默认本地**：采集、转写、匹配、命中片段在本机完成；`--semantic openai` 仅发送转写窗口到你配置的端点，不上传音视频。
- **必须确认许可**：`start` 必须带 `--consent`，或在 config 里 `consent: true`。
- **只处理用户自己的课堂**：不抓取、不下载他人课程。
- **可回放**：时间戳、原话、完整区间音频都在 `hits/hit_NNN/` 里，结论可以人工核对。
- **不做哈希校验**：学习 Skill 不生成 manifest/SHA256SUMS；需要完整性校验时用法律取证 Skill。
- **可发布性**：包内不写死本机路径、用户名、模型位置或密钥；所有路径通过
  CLI 参数 / 环境变量 / 本机缓存提供，`doctor.py` 只在运行时读取本机环境。
- **不做笔记总结**：它只负责“找到并保存那一段”，总结交给用户或别的工具。
- **不是法律/合规工具**：课堂录音的使用请遵守学校规定和当地法律。

## Reference Documents

- [live-mode.md](references/live-mode.md)：实时模式配置、输出、调参、故障排查。
- [output-format.md](references/output-format.md)：实时 `hit.json` 与课后 `result.json` 字段。
- [semantic-mode.md](references/semantic-mode.md)：host 语义判定、知识点契约、输出字段。
- [asr-setup.md](references/asr-setup.md)：转写模型与字幕格式。
- [acceptance-criteria.md](references/acceptance-criteria.md)：比赛提交规范与自检清单。
- [privacy-and-consent.md](references/privacy-and-consent.md)：录音来源、同意与使用边界。

## Examples

### 实时模式：3 步

```bash
python scripts/live_ctl.py doctor
python scripts/live_ctl.py start --config examples/interests.example.json --out out/lecture --consent
python scripts/live_ctl.py watch --out out/lecture
```

命中后：

```text
[命中开始] dp @ 1320.5s 从 01:30 开始，继续抓取完整知识点区间…
  今天我们来学习动态规划的状态转移方程。
[命中完成] dp 01:30 – 06:30（300 秒）
  状态转移方程是动态规划的核心。
  命中目录：out/lecture/hits/hit_001
```

### 课后模式：定位动态规划重点

```bash
python scripts/find_moment.py \
  --transcript examples/lecture-transcript.json \
  --query "老师讲动态规划状态转移方程的那几分钟"
```

预期：完整讨论 `40:00 – 44:48`，核心片段 `41:20 – 42:28`。

### 自测

```bash
python scripts/selftest.py
```

预期输出 `SELFTEST PASS`。

### 打包提交

```bash
python scripts/package.py . --out ../submission/lecture-moment-finder.zip
```

会检查文件数 ≤30、总大小 ≤40MB、`SKILL.md` ≤500 行、扩展名合规，
并自动排除 `__pycache__`、`out/` 等运行产物。