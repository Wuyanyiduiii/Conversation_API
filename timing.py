"""
timing.py — 每轮对话的延迟计时数据结构。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TurnTiming:
    asr_ms: Optional[float] = None
    llm_ttft_ms: Optional[float] = None      # LLM 首 token 延迟
    llm_total_ms: Optional[float] = None     # LLM 完整流结束
    tts_first_chunk_ms: Optional[float] = None  # 第一句 TTS API 耗时
    total_ms: Optional[float] = None         # VAD 结束 → 第一块音频入队

    def report(self) -> str:
        parts: list[str] = []
        if self.asr_ms is not None:
            parts.append(f"ASR收尾: {self.asr_ms:.0f}ms")
        if self.llm_ttft_ms is not None:
            parts.append(f"LLM首字: {self.llm_ttft_ms:.0f}ms")
        if self.llm_total_ms is not None:
            parts.append(f"LLM总计: {self.llm_total_ms:.0f}ms")
        if self.tts_first_chunk_ms is not None:
            parts.append(f"TTS首句: {self.tts_first_chunk_ms:.0f}ms")
        if self.total_ms is not None:
            parts.append(f"端到端: {self.total_ms:.0f}ms")
        return " | ".join(parts) if parts else "(no timing data)"
