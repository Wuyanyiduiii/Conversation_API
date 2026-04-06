"""
conversation_sdk.py — 灵鹦语音对话 SDK。

将 ASR → LLM → TTS 全链路封装为一个有状态的会话对象。
调用方只需提供配置 JSON、录制好的 PCM 音频，即可获得 AI 回复文字与 PCM 音频。

Quick start
-----------
    from conversation_sdk import ConversationSession

    config = {
        "system_prompt": "你是一名热情的咖啡店店员，名叫小美，用轻松口吻与顾客交流。",
        "opening_line":  "您好，欢迎光临！今天想喝点什么？",
        "tts_voice_id":  "female-shaonv",
    }

    session = ConversationSession(config)

    # 1. 获取开场白（文字 + 音频）
    text, audio = session.get_opening()   # audio: 32kHz 16-bit PCM bytes

    # 2. 对话循环
    while True:
        pcm = ...  # 自行录音，16kHz 16-bit mono PCM bytes
        result = session.process_turn(pcm)
        print(result.user_text, result.assistant_text)
        # result.audio: 32kHz 16-bit PCM，可直接播放
        if result.end_flag == 1:
            break
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from asr_client import transcribe_pcm
from config import MAX_HISTORY_TURNS
from llm_client import stream_turn
from timing import TurnTiming
from tts_client import synthesize


@dataclass
class TurnResult:
    """单轮对话的完整结果。"""

    user_text: str
    """ASR 识别出的用户文字。"""

    assistant_text: str
    """AI 回复的完整文字。"""

    audio: bytes
    """AI 回复的 PCM 音频（32 kHz, 16-bit signed, 单声道）。"""

    end_flag: int
    """0 = 对话继续；1 = 对话已结束（AI 判断任务完成或耐心耗尽）。"""

    timing: TurnTiming
    """分段延迟数据，可调用 timing.report() 获取可读字符串。"""

    is_fallback: bool = False
    """True 表示本轮 AI 文字使用了兜底内容，而非模型正常生成。"""


class ConversationSession:
    """
    有状态的语音对话会话。

    会话内部维护对话历史，每次调用 process_turn / process_text_turn 均会
    自动追加到历史中，无需外部管理。

    Parameters
    ----------
    config : dict
        会话配置，必填字段：

        system_prompt  (str)
            AI 角色的系统提示词。
        opening_line   (str)
            AI 的开场白文字（由 get_opening() 合成后播放）。
        tts_voice_id   (str)
            MiniMax 音色 ID，例如 "female-shaonv"、"male-qn-badao"。

        可选字段：

        task               (str)  任务描述，用于让 AI 判断对话是否完成（影响 end_flag）。
        max_history_turns  (int)  保留的对话轮数（1轮 = 1条 user + 1条 assistant），
                                  默认值来自 config.py 中的 MAX_HISTORY_TURNS。
    """

    REQUIRED_FIELDS = ("system_prompt", "opening_line", "tts_voice_id")

    def __init__(self, config: dict) -> None:
        for key in self.REQUIRED_FIELDS:
            if key not in config:
                raise ValueError(f"ConversationSession: config 缺少必填字段 {key!r}")

        self._voice_id: str = config["tts_voice_id"]
        self._scenario: dict = {
            "system_prompt": config["system_prompt"],
            "opening_line":  config["opening_line"],
            "task":          config.get("task", ""),
        }
        self._max_history: int = int(config.get("max_history_turns", MAX_HISTORY_TURNS))
        self._history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def get_opening(self) -> tuple[str, bytes]:
        """
        合成开场白并返回。

        Returns
        -------
        (text, audio)
            text  : 开场白文字字符串。
            audio : 32 kHz, 16-bit signed PCM 单声道字节；opening_line 为空时返回 b""。

        说明
        ----
        调用后开场白会被追加到对话历史（role=assistant），后续 process_turn
        时模型能感知到自己说过的开场白。
        """
        text = self._scenario["opening_line"].strip()
        if not text:
            return "", b""
        audio, _ = synthesize(text, voice_id=self._voice_id)
        self._history.append({"role": "assistant", "content": text})
        return text, audio

    def process_turn(
        self,
        pcm_audio: bytes,
        on_audio_chunk: Callable[[bytes], None] | None = None,
    ) -> TurnResult:
        """
        处理一轮对话（包含 ASR）。

        Parameters
        ----------
        pcm_audio : bytes
            用户语音的原始 PCM 字节（16 kHz, 16-bit signed, 单声道）。
        on_audio_chunk : callable, optional
            每合成好一句 TTS 音频后立即调用，参数为该句 PCM bytes。
            用于流式播放——句 1 完成即开始播放，无需等全部句子合成完毕。

        Returns
        -------
        TurnResult
            若 user_text 为空（静音或识别失败），返回空 TurnResult，
            end_flag=0，audio=b""，不更新历史。
        """
        timing = TurnTiming()
        t0 = time.perf_counter()
        user_text, _ = transcribe_pcm(pcm_audio)
        timing.asr_ms = (time.perf_counter() - t0) * 1000
        return self._run_llm_tts(user_text, timing, on_audio_chunk=on_audio_chunk)

    def process_text_turn(
        self,
        user_text: str,
        asr_ms: float | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
    ) -> TurnResult:
        """
        处理一轮对话（跳过 ASR，直接传入已识别文字）。

        适用于调用方自行完成流式 ASR 的场景（例如 main.py 的 Push-to-Talk 模式），
        避免对同一段音频做二次识别。

        Parameters
        ----------
        user_text       : 已识别的用户文字。
        asr_ms          : 可选，外部 ASR 耗时（ms），填入 timing.asr_ms。
        on_audio_chunk  : 可选回调，每合成好一句 TTS 音频后立即调用，
                          参数为该句的 PCM bytes（32kHz 16-bit）。
                          用于流式播放——无需等全部合成完再播放。

        Returns
        -------
        TurnResult
            result.audio 为所有句子音频的拼接，适合需要完整音频的场景。
            同时若提供了 on_audio_chunk，各句音频在合成完毕后立即回调。
        """
        timing = TurnTiming()
        timing.asr_ms = asr_ms
        return self._run_llm_tts(user_text, timing, on_audio_chunk=on_audio_chunk)

    def reset(self) -> None:
        """清空对话历史，开始全新会话（保留配置）。"""
        self._history.clear()

    @property
    def history(self) -> list[dict]:
        """只读对话历史副本。格式：[{"role": "user"|"assistant", "content": str}, ...]"""
        return list(self._history)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_llm_tts(
        self,
        user_text: str,
        timing: TurnTiming,
        on_audio_chunk: Callable[[bytes], None] | None = None,
    ) -> TurnResult:
        """LLM 流式生成 → 并发 TTS → 按序收集音频。"""
        if not user_text.strip():
            return TurnResult(
                user_text=user_text,
                assistant_text="",
                audio=b"",
                end_flag=0,
                timing=timing,
            )

        t_total = time.perf_counter()
        tts_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts")
        futures: list[Future] = []
        collected: list[str] = []
        is_fallback = False
        end_flag = 0

        for event in stream_turn(
            user_text,
            self._history,
            self._scenario,
            max_history=self._max_history,
        ):
            if event["type"] == "sentence":
                s = event["text"]
                collected.append(s)
                futures.append(tts_pool.submit(synthesize, s, self._voice_id))
                if event.get("is_fallback"):
                    is_fallback = True
            elif event["type"] == "done":
                timing.llm_ttft_ms = event["ttft_ms"]
                timing.llm_total_ms = event["total_ms"]
                end_flag = event["end_flag"]

        tts_pool.shutdown(wait=False)
        assistant_text = "".join(collected)

        # 按原句序收集 PCM，同时触发 on_audio_chunk 回调（用于流式播放）
        audio_chunks: list[bytes] = []
        first = True
        for fut in futures:
            pcm, tts_ms = fut.result()
            if first:
                timing.tts_first_chunk_ms = tts_ms
                timing.total_ms = (time.perf_counter() - t_total) * 1000
                first = False
            audio_chunks.append(pcm)
            if on_audio_chunk is not None:
                on_audio_chunk(pcm)

        if timing.total_ms is None:
            timing.total_ms = (time.perf_counter() - t_total) * 1000

        # 更新 history（严格保持 user/assistant 交替，避免 API 拒绝两连 user）
        self._history.append({"role": "user", "content": user_text})
        self._history.append({
            "role": "assistant",
            "content": assistant_text if not is_fallback else "嗯。",
        })

        return TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            audio=b"".join(audio_chunks),
            end_flag=end_flag,
            timing=timing,
            is_fallback=is_fallback,
        )
