# Conversation_API

低延迟语音对话 SDK。将 ASR → LLM → TTS 全链路封装为一个有状态的会话对象，
你只需提供角色配置，输入录音，即可得到 AI 的语音回复。

---

## 目录

- [工作原理](#工作原理)
- [前置条件](#前置条件)
- [安装](#安装)
- [配置 .env](#配置-env)
- [快速开始](#快速开始)
- [SDK 参考](#sdk-参考)
- [延迟指标](#延迟指标)
- [文件结构](#文件结构)

---

## 工作原理

```
按下空格 → 麦克风录音（同步推流给 ASR）
松开空格 → ASR 收尾（~100-300ms）→ 打印用户文字
          → LLM 流式生成，每完成一句立即提交 TTS
          → TTS 多线程并发合成，首句完成即开始播放
          → 后续句子在播放期间并发完成
```

- **技术栈**：阿里云 DashScope Paraformer（ASR）· DeepSeek（LLM）· MiniMax T2A v2（TTS）
- **模型默认值**：`paraformer-realtime-v2` · `deepseek-chat` · `speech-01-turbo`
- **系统要求**：macOS / Linux（Push-to-Talk 依赖 `termios`）· Python 3.9+

---

## 前置条件

```bash
brew install portaudio   # macOS
```

需要三个平台的 API Key：
- [阿里云 DashScope](https://dashscope.console.aliyun.com/apiKey) — ASR
- [DeepSeek](https://platform.deepseek.com/api_keys) — LLM
- [MiniMax](https://platform.minimaxi.com/user-center/basic-information/interface-key) — TTS

---

## 安装

```bash
git clone <repo-url>
cd Conversation_API
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 配置 .env

复制模板并填入 Key：

```bash
cp .env.example .env
```

`.env` 必填项：

```dotenv
DASHSCOPE_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
MINIMAX_API_KEY=eyJ...
MINIMAX_GROUP_ID=17...
```

模型名称均有默认值，不填即使用默认。**音色 ID 在代码的 config 字典里设置，不在 `.env` 里。**

调整对话历史长度（默认 6 轮）：编辑 `config.py` 中的 `MAX_HISTORY_TURNS`。

---

## 快速开始

```bash
python demo.py
```

`demo.py` 是一个完整的 Push-to-Talk 示例，修改其中的 `CONFIG` 字典即可切换角色：

```python
CONFIG = {
    "system_prompt": (
        "你是热情的咖啡店店员小美，用轻松口吻帮顾客点单。\n"
        '只输出 JSON：{"text": "你的回复，1-2句话", "end": 0}\n'
        "顾客完成点单或说再见时，end 置为 1。"
    ),
    "opening_line": "您好，欢迎光临！今天想喝点什么？",
    "tts_voice_id": "female-shaonv",
}
```

---

## SDK 参考

### 初始化

```python
from conversation_sdk import ConversationSession

session = ConversationSession({
    "system_prompt": "...",   # 必填：AI 角色提示词
    "opening_line":  "...",   # 必填：开场白（空字符串则跳过）
    "tts_voice_id":  "...",   # 必填：MiniMax 音色 ID
    "task":          "...",   # 可选：任务描述，AI 据此判断何时置 end=1
    "max_history_turns": 6,   # 可选：对话历史轮数（默认 6）
})
```

**常用音色**：`female-shaonv`（少女）· `female-yujie`（御姐）· `male-qn-jingying`（精英男）· `male-qn-badao`（霸道男）
**全部音色可查看voice_id_list.md。**

### 方法

```python
# 合成开场白，返回 (文字, PCM音频字节)
text, audio = session.get_opening()

# 方式一：传入录音 PCM，SDK 内部完成 ASR → LLM → TTS
result = session.process_turn(pcm_16k, on_audio_chunk=player.enqueue)

# 方式二：跳过 ASR，直接传已识别文字（适合自行做流式 ASR）
result = session.process_text_turn(user_text, asr_ms=asr_ms,
                                   on_audio_chunk=player.enqueue)

# 重置历史，开始新一轮对话
session.reset()
```

> `on_audio_chunk` 回调：每句 TTS 合成完立即触发，实现逐句流式播放，首句出声更快。

### TurnResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_text` | `str` | ASR 识别出的用户文字 |
| `assistant_text` | `str` | AI 回复的完整文字 |
| `audio` | `bytes` | AI 回复的完整 PCM 音频（32 kHz, 16-bit, mono） |
| `end_flag` | `int` | `0` 继续对话 / `1` 对话结束 |
| `timing` | `TurnTiming` | 调用 `timing.report()` 打印分段耗时 |
| `is_fallback` | `bool` | `True` 表示 AI 使用了兜底回复 |

### 音频格式

| 方向 | 格式 |
|------|------|
| `process_turn` 输入 | 16 kHz · 16-bit signed · mono |
| `result.audio` / `on_audio_chunk` 输出 | 32 kHz · 16-bit signed · mono |

---

## 延迟指标

每轮对话后 `timing.report()` 输出：

```
ASR收尾: 187ms | LLM首字: 420ms | LLM总计: 1350ms | TTS首句: 310ms | 端到端: 780ms
```

| 字段 | 含义 |
|------|------|
| `asr_ms` | 停止录音 → ASR 返回文字 |
| `llm_ttft_ms` | ASR 结束 → LLM 首 token |
| `llm_total_ms` | LLM 流完整结束 |
| `tts_first_chunk_ms` | 第一句 TTS API 耗时 |
| `total_ms` | 停止录音 → 第一块音频入队（端到端） |

---

## 文件结构

```
Conversation_API/
├── demo.py               # 快速开始示例（修改 CONFIG 即可使用）
├── conversation_sdk.py   # SDK 核心：ConversationSession
├── asr_client.py         # DashScope Paraformer 流式 ASR
├── llm_client.py         # DeepSeek 流式 LLM，含句子切分
├── tts_client.py         # MiniMax TTS，输出 32kHz PCM
├── audio_io.py           # Push-to-Talk 录音 + 顺序播放队列
├── timing.py             # 延迟计时数据结构
├── config.py             # 从 .env 加载 API Key 与模型配置
├── requirements.txt      # Python 依赖
├── .env.example          # 配置模板
└── .gitignore
```
