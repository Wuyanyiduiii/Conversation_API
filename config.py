"""
config.py — 从 .env 加载所有配置项。

在此文件顶部可直接修改 MAX_HISTORY_TURNS 调节对话历史长度。
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── ASR (阴云 DashScope Paraformer) ────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY: str = os.environ["DASHSCOPE_API_KEY"]
ASR_MODEL: str = os.environ.get("ASR_MODEL", "paraformer-realtime-v2")

# ── LLM (DeepSeek) ───────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.environ["DEEPSEEK_API_KEY"]
LLM_MODEL: str = os.environ.get("LLM_MODEL", "deepseek-chat")

# ── TTS (MiniMax) ─────────────────────────────────────────────────────────────
MINIMAX_API_KEY: str = os.environ["MINIMAX_API_KEY"]
MINIMAX_GROUP_ID: str = os.environ["MINIMAX_GROUP_ID"]
TTS_MODEL: str = os.environ.get("TTS_MODEL", "speech-01-turbo")
TTS_VOICE_ID: str = os.environ["TTS_VOICE_ID"]

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_FILE: str = os.environ.get("SYSTEM_PROMPT_FILE", "test_prompt.md")

# ── Conversation History ──────────────────────────────────────────────────────
# 保留的对话轮数（1 轮 = 1 条 user + 1 条 assistant）。
# 调大 → 上下文更连贯；调小 → 延迟更低、token 消耗更少。
MAX_HISTORY_TURNS: int = 6
