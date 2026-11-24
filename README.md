# lecture-moment-finder · 课堂回放定位

按你的兴趣清单，在课堂/会议里**只保存真正讲到的那一段**：完整区间音频 + 屏幕录像 + 原话转写；课后一句话定位"老师讲 XX 的那几分钟"，并切出 30–90 秒核心片段。

不做平滑的会议纪要，不做课程总结。**别人帮你记下来，它帮你证出来。**

## 它怎么工作

```text
课前：interests.json（你关心的问题 / 项目术语 / 薄弱知识点）
  ↓
实时：本机录音（系统声音或麦克风）+ 录屏，faster-whisper 本地转写
  ↓
判定：每 20–30 秒把最近 45 秒转写交给 LLM，判 teach / mention / none
      host = 宿主 agent 自己的模型（本机文件请求/响应）
      openai = 任意 OpenAI 兼容端点（仅在显式指定时发送转写窗口）
  ↓
命中：只有 teach 才开区间；不按关键词触发
  ↓
跟踪：后续窗口继续 teach 就延长；连续两个窗口不再教 → 收尾
  ↓
保存：hits/hit_NNN/ ← 完整区间音频、屏幕录像、转写、原话
```

课后模式：`scripts/find_moment.py` 对整段转写做定位，返回完整讨论区间 + 核心 30–90 秒，并可切片。

### 宿主 retry（二轮检索）

首轮候选判不出 `teach` 时，不直接降级：宿主可返回 `retry_terms`（ASR 错写/同义表述，例如"高斯"被转成"公因公式"），脚本据此重建候选池再判一轮；仍无 `teach` 才走 AI 补讲（明确标注"非课堂原话、无时间戳"）。

```bash
python scripts/find_moment.py --transcript lecture.json --query "..." \
  --semantic host --judge-request-out req.json
python scripts/find_moment.py ... --retry-terms "散度定理,高斯公式,三重积分" \
  --judge-response resp2.json
```

## 快速开始

```bash
python scripts/doctor.py                 # 依赖与设备检查（ffmpeg/PyAV/faster-whisper/模型）
python scripts/selftest.py               # 离线 + 实时完整区间自测，期望 SELFTEST PASS
python scripts/live_ctl.py start --config interests.json --out out --consent
python scripts/live_ctl.py watch --out out
python scripts/live_ctl.py stop --out out
```

课后定位：

```bash
python scripts/find_moment.py --transcript examples/lecture-transcript.json \
  --query "老师讲动态规划状态转移方程的那几分钟" --media lecture.mp4
```

Python 3.10–3.12（3.13 移除了 `audioop`，不支持）。依赖统一由 `scripts/deps.py` 解析；没有 ffmpeg 时自动用 PyAV 切片。

## 评测（如实口径）

2026 高等微积分 2 习题集 55 个知识点、69 道题、34 个录播（68.7 小时）；转写为本地 Paraformer，判定为宿主 LLM。指标：C＝完整讲解且区间对，P＝提及/部分，W＝段错，N＝没找到；严格召回＝C/55，可用＝(C+P)/55。

| 版本 | 判定 | 严格召回 | 可用召回 | teach 桶严格精度 |
|---|---|---|---|---|
| v1（首版，已上架审核） | 34 teach / 18 mention / 3 N | 58.18% | 90.91% | 94.12% |
| v1.1（本仓库，含 retry） | 36 teach / 16 mention / 3 N | **60.00%** | **90.91%** | 91.67% |

限制（必须一起读）：
- retry 只重测了 21 个非 teach 点，新翻盘的 2 点（kp49 M-判别 / kp51 幂级数逐项积分）为单一审计者评分，非独立复核；按题 69 题成功率未重算。
- 不宣称 90%+ 严格召回；不宣称 CER/WER（无人工逐字转写）；不宣称用户满意度（无真实用户样本）。
- 自由"方法/解法"问法（如"题 7 怎么算"）依赖 retry 二轮；已验证一例：命中散度定理（Gauss）讲解 29:55–30:50。

## 隐私

默认本机运行，不上传音视频。只有显式 `--semantic openai` 时，才把最近一段转写窗口发送到你自己指定的端点。采集前需 `--consent`；请遵守当地录音/录屏法规与课堂/会议纪律。

## 目录

```text
SKILL.md                 # Skill 说明与触发词
scripts/                 # 采集、判定、定位、切片、自测、打包
references/              # 实时模式、隐私、验收标准
examples/                # 关心清单与转写样例
```

## License

尚未指定（TBD）。如需 MIT / Apache-2.0，请提 issue 或联系作者。