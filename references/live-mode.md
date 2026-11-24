# 实时模式（Live Mode）

实时模式是 Skill 的课堂采集版：上课前写下关心的问题，启动后自动录音/录屏，
实时转写并交给 LLM 判断“老师此刻是不是在教这个知识点”；只有 `teach` 才命中，
命中后继续跟踪到老师讲完，再保存完整区间的音频、屏幕录像和原话（例如 01:30–06:30
全程）。**触发不依赖关键词**；没有 LLM 后端时只保留手动 VAR。

## 1. 依赖

实时模式需要本机 Python 3.10+ 以及：

| 包 | 用途 |
|---|---|
| `pyaudiowpatch` | Windows WASAPI 系统声音环回 + 麦克风 |
| `mss` + `Pillow` | 屏幕采集 |
| `av`（PyAV） | 屏幕录像编码（不需要外部 ffmpeg） |
| `numpy` | 音频转 float32 给 faster-whisper |
| `faster-whisper` | 本地实时转写 |

如果这些已经装在另一个 Python 环境里，直接用：

```bash
"<path-to-python>" scripts/live_ctl.py doctor
```

不需要在当前 Python 里重复安装。`live_ctl.py start` 支持 `--python` 指定解释器。

## 2. 配置文件

```json
{
  "session_id": "lecture-2026-01-10",
  "title": "数据结构 第 7 讲",
  "consent": false,
  "save_transcript": true,
  "interests": [{
    "id": "dp",
    "question": "老师讲动态规划状态转移方程的那几分钟",
    "definition": "写出状态定义、转移方程、边界和计算顺序，并用例题说明",
    "keywords": ["动态规划", "状态转移方程"]
  }],
  "semantic": {
    "enabled": true,
    "backend": "host",
    "window_seconds": 45.0,
    "interval_seconds": 25.0,
    "min_new_chars": 80,
    "min_confidence": 0.6,
    "max_pending": 2,
    "max_interests": 40,
    "timeout_seconds": 60.0
  },
  "capture": {
    "source": "loopback",
    "screen": true,
    "fps": 1.0,
    "video_fps": 5.0,
    "video_width": 1280,
    "video_crf": 28,
    "monitor": 1
  },
  "asr": {
    "model": "base",
    "language": "zh",
    "device": "cpu",
    "chunk_seconds": 10.0,
    "step_seconds": 3.0,
    "vad_filter": true
  },
  "trigger": {
    "ring_seconds": 300.0,
    "min_span": 45.0,
    "max_span": 600.0,
    "semantic_grace_seconds": 90.0,
    "manual_span_seconds": 120.0,
    "max_gap": 20.0
  }
}
```

### 字段说明

| 字段 | 说明 |
|---|---|
| `session_id` | 会话 ID，输出目录使用 |
| `consent` | 是否已确认录音/录屏许可；也可用 `--consent` |
| `save_transcript` | 是否保留实时转写；隐私优先时设 false |
| `interests[].question` | 一句具体问题，作为语义判定的主依据 |
| `interests[].definition` | 可选；写清“讲到什么才算教”，最能减少误判 |
| `interests[].keywords` | 只作 LLM 提示，**不参与触发** |
| `semantic.enabled` | false=只手动 VAR，不做自动判定 |
| `semantic.backend` | `host`=宿主 agent 模型；`openai`=OpenAI 兼容端点 |
| `semantic.window_seconds` | 每次判定回看多少秒转写，默认 45 |
| `semantic.interval_seconds` | 两次判定的最小间隔，默认 25 |
| `semantic.min_new_chars` | 至少新增多少字才发下一次判定 |
| `semantic.min_confidence` | 低于该值的 `teach` 只算 `mention` |
| `semantic.max_pending` | host 模式最多挂起几个未判定请求 |
| `semantic.max_interests` | 单次请求最多带多少个兴趣点 |
| `semantic.timeout_seconds` | openai 请求超时 |
| `capture.source` | `loopback` / `mic` / `both` |
| `capture.screen` | 是否录屏 |
| `capture.fps` | 截图/前段帧采样率，1–2 足够 |
| `capture.video_fps` | 录屏帧率，默认 5 |
| `capture.video_width` | 录屏宽度，默认 1280 |
| `capture.video_crf` | 录屏质量（CRF 越大文件越小），默认 28 |
| `capture.monitor` | 录制哪个显示器，默认 1（主屏） |
| `asr.model` | `tiny` / `base` / `small` / `none`（none 只测采集） |
| `asr.chunk_seconds` | 每次转写的音频窗口，默认 10 秒 |
| `asr.step_seconds` | 多久跑一次，默认 3 秒 |
| `asr.vad_filter` | 是否先做 VAD；失败会自动回退 |
| `trigger.ring_seconds` | 音频/屏幕环形缓冲秒数；决定最多能向前回溯多久 |
| `trigger.min_span` | 命中区间最短秒数 |
| `trigger.max_span` | 单个区间最长秒数；超长知识点自动分段 |
| `trigger.semantic_grace_seconds` | 多久没有新的 `teach` 判定就收尾 |
| `trigger.manual_span_seconds` | 手动 VAR 抓取窗口，默认 120 秒 |
| `trigger.max_gap` | 相邻两句间隔超过多少秒视为停顿 |

> 旧关键词模式的 `min_score`、`min_keyword_hits`、`consecutive_hits`、
> `trigger.window_seconds`、`trigger.end_grace_seconds`、`trigger.rel_threshold`、
> `trigger.bridge` 仍然兼容，但只在 `semantic.enabled=false` 时参与命中。

## 3. 命令

```bash
# 依赖 + 设备
python scripts/live_ctl.py doctor
python scripts/live_ctl.py devices

# 启动（后台）
python scripts/live_ctl.py start \
  --config examples/interests.example.json \
  --out out/lecture-2026-01-10 \
  --consent

# 使用指定 Python（依赖装在另一个环境时）
python scripts/live_ctl.py start \
  --config examples/interests.example.json \
  --out out/lecture-2026-01-10 \
  --consent \
  --python "<path-to-python>"

# 前台调试
python scripts/live_ctl.py start ... --foreground --duration 30

# 状态 / 监听命中 / 停止
python scripts/live_ctl.py status --out out/lecture-2026-01-10
python scripts/live_ctl.py watch --out out/lecture-2026-01-10
python scripts/live_ctl.py stop --out out/lecture-2026-01-10

# 手动 VAR：抓当前时刻前后各 60 秒
python scripts/live_ctl.py var --out out/lecture-2026-01-10
python scripts/live_ctl.py var --out out/lecture-2026-01-10 --at 320.5 --note "这里没听懂"

# 强制只用 LLM（不回退关键词）
python scripts/live_ctl.py start ... --semantic host
python scripts/live_ctl.py start ... --semantic openai
```

### 3.1 CodeArts 直接读转写判定

`semantic.backend="host"` 时，引擎不会调用模型。它把最近 45 秒转写和兴趣定义
写成 `judge/requests/req_NNNN.json`，并把待判定列表写到 `judge/pending.json`。
**CodeArts 直接用 Read 工具读这些文件，用宿主模型判定，用 Write 工具写回**，
不需要经过任何“判定通道”：

1. 读 `out/.../judge/pending.json`：
   ```json
   {"pending":[{"request_id":"req_0001","path":".../requests/req_0001.json",
     "window":{"start":0,"end":45},"interests":["dp"]}]}
   ```
2. 打开 `path`，用 `request.interests` 和 `request.transcript` 判定；需要更长上下文时
   直接读同目录的 `transcript.jsonl`，判定窗口仍以 `request.window` 为准。
3. 写 `out/.../judge/responses/req_0001.json`：
   ```json
   {"judgments":[{"interest_id":"dp","level":"teach","confidence":0.92,
     "start":12.0,"end":38.0,"quote":"原话","reason":"正在写转移方程"}]}
   ```
4. 每 20–30 秒重复；没有 pending 就等下一轮，**不要用关键词代替**。

`judge_live.py` 仍然保留，但只是可选的读写助手：

```bash
python scripts/judge_live.py next --out ... --timeout 60   # 等请求并打印
python scripts/judge_live.py submit --out ... --request req_0001 --file resp.json
```

引擎下一轮读取 `judge/responses/`，把 `teach` 变成 `hit_open`，
`mention`/`none` 只更新“不再教”的计数。

### 3.2 OpenAI 兼容端点

```bash
export LECTURE_LLM_BASE_URL="http://127.0.0.1:PORT/v1"
export LECTURE_LLM_MODEL="your-model"
export LECTURE_LLM_API_KEY="..."      # 可选
python scripts/live_ctl.py start ... --semantic openai
```

也可写到 `semantic.openai.base_url/model/api_key`。host 模式只在本机读写文件；
openai 模式会把转写窗口发到你配置的端点。两种模式都不会回退关键词。

### 3.3 悬浮 HUD（可选）

`hud.py` 是不依赖第三方包的 tkinter 悬浮球，负责“问兴趣 → 开始 → 手动 VAR”：

```bash
python scripts/hud.py --config out/interests.json --out out/lecture --ask-interests
```

- `--ask-interests`：先弹出兴趣设置，每行 `问题 | 讲到什么才算教`；确认录音/录屏许可后启动引擎。
- HUD 置顶、可拖动；显示运行时长、命中数、转写行数、语义后端、正在抓取的主题。
- **Enter**（HUD 获得焦点时）或全局 **F9 / Ctrl+Alt+V**：写 `MANUAL.json`，引擎抓前后各 60 秒。
- “兴趣”按钮可编辑 JSON；保存后停止并重新开始生效。
- “停止”写 `STOP`，引擎收尾；关闭 HUD 不会停止引擎。
- 已有会话：`--no-start` 只显示 HUD，附着到正在运行的引擎。
- HUD 的 Python 没有采集依赖时，加 `--python "<path-to-python>"`。
- 没有桌面/无法弹窗的环境不要用 HUD，直接用 `live_ctl.py`。

## 4. 输出

```text
out/lecture-2026-01-10/
├── status.json
├── transcript.jsonl
├── hits.jsonl          # hit_open / hit_close 事件流
├── session.json
├── engine.log
├── judge/              # host 模式：pending.json / requests / responses / applied
├── MANUAL.json         # 手动 VAR 的临时标记（引擎消费后删除）
└── hits/
    └── hit_001/
        ├── audio.wav    # 完整知识点区间
        ├── screen.mp4   # 同一区间的屏幕录像（5fps / 1280 宽）
        ├── transcript.txt
        └── hit.json     # 区间、核心片段、原话、结束原因
```

没有 PyAV 时，`screen.mp4` 会自动回退为 `screens/` 关键帧截图（最多 60 张）。

`status.json` 是给 CodeArts agent / HUD 轮询的：

```json
{
  "state": "recording",
  "elapsed": 1320.5,
  "hits": 2,
  "last_text": "状态转移方程是动态规划的核心。",
  "semantic": {
    "enabled": true, "backend": "host", "model": "host-agent",
    "requests": 12, "pending": 1, "last_error": ""
  },
  "last_semantic": {
    "request_id": "req_0012", "window_start": 1275.0, "window_end": 1320.0,
    "teach": ["dp"], "mention": [], "none": []
  },
  "open_hits": [
    {
      "hit_id": "hit_003",
      "interest_id": "dp",
      "at": 1300.0,
      "span_start_text": "21:40",
      "duration": 20.5
    }
  ],
  "last_hit": {
    "interest_id": "dp",
    "span_start_text": "18:20",
    "span_end_text": "20:05",
    "duration": 105.0,
    "core_start_text": "18:55",
    "core_end_text": "19:50",
    "key_quote": "状态定义是骨架，转移方程是肌肉。",
    "dir": "out/lecture/hits/hit_002"
  }
}
```

`hits.jsonl` 有两类事件：`hit_open`（开始抓取）与 `hit_close`（收尾完成，
带完整区间、核心片段和 `dir`）。

## 5. 与完整引擎的功能对齐

| 完整引擎 | Skill 实时模式 | 状态 |
|---|---|---|
| 系统声音环回 + 麦克风 | `capture.source=loopback/mic/both` | ✅ |
| 屏幕采集 | `capture.screen` + `ScreenRing` | ✅ |
| 实时 ASR | faster-whisper 分块 + 重叠拼接 | ✅ |
| 课前关注清单 | `interests.json` | ✅ |
| 语义/兴趣触发 | 宿主模型或 OpenAI 兼容端点判 `teach/mention/none` | ✅ |
| 手动 VAR | `live_ctl.py var`，前后各 60 秒 | ✅ |
| 完整知识点区间 | 命中后跟踪 + 自适应区间（不是固定 90 秒） | ✅ |
| 屏幕录屏 | `screen.mp4`，覆盖完整区间（PyAV，无需 ffmpeg） | ✅ |
| 可回放材料 | `audio.wav` / `screen.mp4` / `transcript.txt` / `hit.json` | ✅ |
| 证据包 + SHA256 | 不在学习 Skill；见法律取证 Skill | — |
| 课后 Find Moment | `find_moment.py` | ✅ |
| 同意/本地优先 | `--consent`；默认本地，`--semantic openai` 仅外发转写窗口 | ✅ |
| 说话人分离 | 未实现 | ⬜ P1 |
| 高清补录（HD） | 未实现 | ⬜ P1 |
| 悬浮 HUD | `hud.py` 置顶悬浮球 + 手动 VAR；副屏/8 寸屏未验证 | ✅ / 副屏 ⬜ |
| 承诺账本 | 不适用于课堂 | — |

## 6. 调参

| 现象 | 调整 |
|---|---|
| 命中太频繁 | 提高 `semantic.min_confidence`、缩小 `window_seconds`、把 `definition` 写具体 |
| 总是漏 | 降低 `min_confidence`、增大 `window_seconds`、补充 `definition` |
| 判定太慢/积压 | 增大 `interval_seconds`、减小 `max_interests`；host 改 openai |
| 转写太慢 | `model: "tiny"`，或增大 `step_seconds` |
| 转写太碎 | 增大 `chunk_seconds`，或 `model: "small"` |
| 繁体输出 | 保留 `initial_prompt` 的“请用简体中文转写” |
| 区间太短/提前收尾 | 提高 `semantic_grace_seconds`、提高 `max_gap` |
| 区间太长 | 降低 `semantic_grace_seconds`、降低 `max_span` |
| 一个知识点被拆成多段 | 提高 `max_span`、提高 `semantic_grace_seconds` |
| 只想测试采集 | `asr.model: "none"` |

## 7. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `pyaudiowpatch` 缺失 | 当前 Python 没装依赖 | `--python` 指向另一个装好依赖的环境 |
| 没有环回设备 | 系统输出不是 WASAPI / 被独占 | `devices` 查看；改用 `mic` 或 `both` |
| `doctor` 正常但没转写 | 音频太轻 / 窗口没覆盖语音 | 检查 `status.last_text`；降低 `vad_rms` |
| 命中一直不触发 | 没有 LLM 后端 / 判定不是 `teach` | 看 `status.semantic.last_error`；跑 `judge_live.py status`；确认宿主循环在跑 |
| host 请求堆积 | 宿主 agent 没在跑 | 启动 `judge_live.py next` 循环；或改用 `--semantic openai` |
| 判定总是 `none` | `definition` 太抽象 / 窗口太短 | 把定义写具体；增大 `window_seconds` |
| 提示 `initial_prompt` | 空白音频导致 Whisper 复读提示词 | 代码已过滤；仍出现就降低 `vad_rms` |
| 后台进程被父进程带走 | CodeArts 使用 job object | 改用 `--foreground --duration`，或从独立终端启动 |
| 录屏/截图失败 | 无桌面会话/权限 | 设 `capture.screen=false`，只用音频 |
| 没有 `screen.mp4`，只有 `screens/` | PyAV 不可用或录屏启动失败 | `python -c "import av"` 检查；`pip install av` |

## 8. 已知限制

- 分块 ASR 在句首/句尾会有少量误差；命中片段以 `audio.wav` + 原话为准。
- 说话人分离未实现；在线课通常只有老师一路，问题不大。
- 语义判定依赖 LLM 后端：host 模式需要宿主 agent 持续跑判定循环，openai 模式需要端点可用。
  后端不可用时不会回退关键词，只会停止自动命中；手动 VAR 仍可用。
- host 模式命中会晚一个判定周期（默认 25 秒 + 宿主模型耗时）；需要更低延迟用 openai。
- 屏幕录像默认 5fps、1280 宽，用于回放定位，不是高清录屏；没有 PyAV 时回退为截图。
- 实时模式依赖本机设备；评委/网页环境请用 `simulate` 回放演示。
- 超长知识点（超过 `max_span`）会自动分成前后两段，分段处不重复、不丢内容。