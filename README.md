# Conversation_API

评估 ASR–LLM–TTS 最优组合，构建低延迟、拟人化的语音对话系统。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [前置条件](#前置条件)
- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [SDK 集成](#sdk-集成)
- [对话场景](#对话场景)
- [快速切换模型](#快速切换模型)
- [延迟指标](#延迟指标)
- [文件结构](#文件结构)

---

## 项目概述

Conversation_API 是一个**全链路语音对话 Pipeline**，用于测试和评估不同 ASR（语音识别）、LLM（大语言模型）、TTS（语音合成）服务组合在实际对话场景中的端到端延迟与效果。

核心特性：

- **Push-to-Talk 录音**：按空格键开始/停止，操作直观，无需唤醒词
- **流式 ASR**：录音与识别同步进行，按下停止键后只需等 100–300ms 即可拿到识别结果
- **流式 LLM + 并发 TTS**：LLM 每生成一个完整句子，立即并行发起 TTS 合成，大幅降低首播延迟
- **顺序播放**：多句 TTS 并发合成后按原顺序入队，保证语义连贯
- **精确延迟统计**：每轮对话自动输出 ASR 收尾、LLM 首 token、TTS 首句、端到端等分段耗时
- **可配置历史轮数**：可在 `config.py` 中调节 `MAX_HISTORY_TURNS`，权衡上下文连贯性与延迟

---

## 系统架构

```
用户按空格 → 麦克风录音
      ↓（PCM 帧实时推流）
  DashScope Paraformer（流式 ASR）
      ↓（用户文字）
  DeepSeek V3/Chat（流式 LLM，JSON 输出）
      ↓（按句切分，并发提交）
  MiniMax TTS（多线程并发合成）
      ↓（PCM 音频，顺序入队）
  扬声器播放（AudioPlayer）
```

**流水线关键优化点**：
- 录音与 ASR 推流同步，避免录完再提交的等待
- LLM 边生成边切句，首句 TTS 在 LLM 输出完毕前即开始合成
- TTS 以 4 个线程并发处理多句，最慢句也在其他句播放时完成

---

## 技术栈

| 模块 | 服务 / 库 | 默认模型 |
|------|----------|---------|
| ASR  | 阿里云 DashScope Paraformer | `paraformer-realtime-v2` |
| LLM  | DeepSeek（OpenAI 兼容接口） | `deepseek-chat` |
| TTS  | MiniMax T2A v2 | `speech-01-turbo` |
| 音频录制/播放 | sounddevice + numpy | — |
| 环境变量管理 | python-dotenv | — |

---

## 前置条件

- Python 3.10+
- macOS（录音模块依赖 `termios`/`tty`，仅限 Unix-like 系统）
- PortAudio（sounddevice 底层依赖）

```bash
brew install portaudio
```

- 三个平台的 API Key：
  - [阿里云 DashScope](https://dashscope.aliyun.com/) — ASR
  - [DeepSeek](https://platform.deepseek.com/) — LLM
  - [MiniMax](https://www.minimaxi.com/) — TTS

---

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd Conversation_API

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 配置

在项目根目录创建 `.env` 文件，填入以下内容：

```dotenv
# ── ASR（阿里云 DashScope）──────────────────────────────
DASHSCOPE_API_KEY=your_dashscope_api_key
ASR_MODEL=paraformer-realtime-v2        # 可选，默认值如左

# ── LLM（DeepSeek）──────────────────────────────────────
DEEPSEEK_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-chat                  # 可选，默认值如左

# ── TTS（MiniMax）────────────────────────────────────────
MINIMAX_API_KEY=your_minimax_api_key
MINIMAX_GROUP_ID=your_minimax_group_id
TTS_MODEL=speech-01-turbo                # 可选，默认值如左
TTS_VOICE_ID=your_voice_id               # 必填，音色 ID

# ── System Prompt ────────────────────────────────────────
SYSTEM_PROMPT_FILE=test_prompt.md        # 可选，默认值如左
```

> `TTS_VOICE_ID` 为必填项，可在 MiniMax 控制台的音色列表中获取。

**对话历史长度**可在 `config.py` 中直接修改：

```python
MAX_HISTORY_TURNS: int = 6  # 1 轮 = 1 条 user + 1 条 assistant
```

---

## 使用方法

```bash
python main.py
```

程序启动后会依次：

1. **显示当前模型配置**（ASR / LLM / TTS / 音色）
2. **列出可用对话场景**，输入编号选择（回车默认选 1）
3. **AI 播放开场白**
4. 进入对话循环：
   - 按 `空格` 开始录音
   - 说话
   - 再按 `空格` 停止，系统自动处理并播放 AI 回复
   - 重复上述步骤
5. 按 `Ctrl+C` 随时退出；或当 AI 判断对话自然结束时自动退出

每轮对话结束后，终端会输出分段延迟统计，例如：

```
[耗时] ASR收尾: 187ms | LLM首字: 420ms | LLM总计: 1350ms | TTS首句: 310ms | 端到端: 780ms
```

---

## SDK 集成

除直接运行 `main.py` 外，您可以将核心对话能力作为 SDK 嵌入自己的程序。
SDK 以一个 JSON 配置初始化，输入 PCM 音频，返回 AI 回复文字与 PCM 音频，录音、界面逻辑完全由调用方掌控。

### 快速开始

参见 `demo.py`——约 60 行代码的完整 Push-to-Talk 示例：

```bash
python demo.py
```

核心代码：

```python
from conversation_sdk import ConversationSession
from audio_io import AudioPlayer, record_push_to_talk

config = {
    "system_prompt": "你是热情的咖啡店店员小美，用轻松口吻帮顾客点单。\n"
                     '只输出 JSON：{"text": "你的回复", "end": 0}',
    "opening_line":  "您好，欢迎光临！今天想喝点什么？",
    "tts_voice_id":  "female-shaonv",
}

session = ConversationSession(config)
player  = AudioPlayer()

text, audio = session.get_opening()   # 合成开场白
player.enqueue(audio)
player.wait_done()

while True:
    pcm    = record_push_to_talk()     # 调用方负责录音
    result = session.process_turn(pcm) # SDK 完成 ASR+LLM+TTS
    player.enqueue(result.audio)
    if result.end_flag == 1:
        player.wait_done()
        break
```

### Config 字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `system_prompt` | ✅ | AI 角色的系统提示词 |
| `opening_line` | ✅ | AI 开场白（空字符串则跳过） |
| `tts_voice_id` | ✅ | MiniMax 音色 ID，例如 `"female-shaonv"`、`"male-qn-badao"` |
| `task` | — | 任务描述，帮助 AI 判断对话何时结束（影响 `end_flag`） |
| `max_history_turns` | — | 保留的对话轮数，默认 6（来自 `config.py`） |

### 方法说明

| 方法 | 说明 |
|------|------|
| `get_opening()` | 合成开场白，返回 `(text: str, audio: bytes)` |
| `process_turn(pcm_audio)` | 传入 16kHz PCM，SDK 内部依次执行 ASR→LLM→TTS，返回 `TurnResult` |
| `process_text_turn(user_text, asr_ms=None, on_audio_chunk=None)` | 跳过 ASR，直接传已识别文字，适合自行做流式 ASR 的场景 |
| `reset()` | 清空对话历史，保留配置 |
| `session.history` | 只读属性，返回 `[{"role": ..., "content": ...}, ...]` 副本 |

### TurnResult 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_text` | `str` | ASR 识别出的用户文字 |
| `assistant_text` | `str` | AI 回复的完整文字 |
| `audio` | `bytes` | AI 回复的 PCM 音频（32 kHz, 16-bit, mono） |
| `end_flag` | `int` | `0`=继续对话，`1`=对话已结束 |
| `timing` | `TurnTiming` | 分段延迟，调用 `timing.report()` 获取可读字符串 |
| `is_fallback` | `bool` | `True` 表示 AI 使用了兜底回复而非正常生成 |

### 音频格式约定

| 方向 | 格式 |
|------|------|
| `process_turn` 输入 | 16 kHz, 16-bit signed, mono PCM |
| `result.audio` 输出 | 32 kHz, 16-bit signed, mono PCM |

### 流式播放回调

对低延迟要求较高时，可使用 `process_text_turn` 的 `on_audio_chunk` 参数——每合成完一句 TTS 立即回调，无需等全部完成，`main.py` 采用此模式：

```python
result = session.process_text_turn(
    user_text,
    asr_ms=asr_ms,
    on_audio_chunk=player.enqueue,  # 每句合成完即立即入队
)
# result.audio 同时包含所有句子拼接后的完整音频
```

---

## 对话场景

场景数据存储在 `test.json` 中，目前内置 6 个汽车零售销售训练场景：

| 编号 | 标题 | 场景描述 | 学习目标 | 客户音色 |
|------|------|---------|---------|---------|
| 1 | 破冰迎宾 | 展厅初次接待犹豫型客户 | 开场白与需求探寻技巧 | `male-qn-jingying` |
| 2 | 需求深挖 | 识别家庭用户的隐性痛点 | SPIN 法则深层需求挖掘 | `female-yujie` |
| 3 | 动态试驾 | 道路实测中的价值传递 | 将车辆性能转化为客户利益 | `male-qn-jingying` |
| 4 | 异议攻坚 | 化解价格与保值率顾虑 | 价格异议处理与价值重塑 | `male-qn-badao` |
| 5 | 竞品狙击 | 直面强势竞品的参数对比 | 客观分析竞品并突出差异化优势 | `male-qn-jingying` |
| 6 | 缔结成交 | 高压下的促单与签约 | 识别购买信号并果断促成交易 | `male-qn-badao` |

每个场景包含：角色背景、对话任务、关键产品信息、AI 客户的耐心值机制（失误扣分、命中加分），以及动态情绪阶段设定。

**自定义场景**：直接编辑 `test.json`，按现有格式新增对象即可。必填字段：`title`、`subtitle`、`user_character`、`background`、`task`、`system_prompt`、`opening_line`、`tts_voice_id`。

---

## 快速切换模型

无需修改 `.env`，直接在 `main.py` 顶部的覆盖变量赋值即可：

```python
ASR_MODEL_OVERRIDE: str | None = "paraformer-realtime-v2"
LLM_MODEL_OVERRIDE: str | None = "deepseek-reasoner"
TTS_MODEL_OVERRIDE: str | None = "speech-01-hd"
```

设为 `None` 则使用 `.env` 中的配置。此设计便于 A/B 对比不同模型的延迟与效果。

---

## 延迟指标

`timing.py` 定义了每轮对话的计时结构 `TurnTiming`，包含以下维度：

| 字段 | 含义 |
|------|------|
| `asr_ms` | Stop 键按下 → ASR 返回最终文字的耗时 |
| `llm_ttft_ms` | ASR 结束 → LLM 流中首 token 到达的耗时 |
| `llm_total_ms` | LLM 流完整结束的耗时 |
| `tts_first_chunk_ms` | 第一句 TTS API 调用耗时 |
| `total_ms` | Stop 键按下 → 第一块 PCM 音频入队的端到端耗时 |

---

## 文件结构

```
Conversation_API/
├── main.py               # 主入口：场景选择、Push-to-Talk 对话循环、模型覆盖配置
├── demo.py               # SDK 使用示例：最简 Push-to-Talk 程序（约 60 行）
├── conversation_sdk.py   # 核心 SDK：ConversationSession 类，封装 ASR→LLM→TTS 全链路
├── pipeline.py           # 旧版独立流水线（保留作参考）
├── config.py             # 从 .env 加载所有配置项
├── asr_client.py         # DashScope Paraformer 流式 ASR 封装
├── llm_client.py         # DeepSeek 流式对话封装，含句子切分逻辑
├── tts_client.py         # MiniMax TTS HTTP 封装，输出 32kHz PCM
├── audio_io.py           # Push-to-Talk 录音 + AudioPlayer 顺序播放
├── timing.py             # 每轮对话延迟数据结构与格式化输出
├── test.json             # 内置对话场景数据（含 tts_voice_id）
├── test_prompt.md        # 默认 System Prompt（可通过 .env 替换）
└── requirements.txt      # Python 依赖列表
```
