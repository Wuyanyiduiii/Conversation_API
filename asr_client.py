"""
asr_client.py — 阿里云 DashScope Paraformer ASR 封装。

支持两种模式：
- 流式 (streaming)：录音时同步推流给 ASR，stop 后极短时间内返回结果（~100-300ms）。
- 批量 (batch)：录完再提交，保留作调试备用。
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import wave
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from config import ASR_MODEL, DASHSCOPE_API_KEY

dashscope.api_key = DASHSCOPE_API_KEY


# ─── 流式 ASR ─────────────────────────────────────────────────────────────────

class _StreamCallback(RecognitionCallback):
    """累积已完结句子；on_close 后 transcript 属性即为完整转录。"""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._partial: str = ""
        self._closed = threading.Event()
        self.error: str | None = None

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        self._closed.set()

    def on_error(self, result: RecognitionResult) -> None:
        self.error = str(result)
        self._closed.set()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not sentence:
            return
        # 流式回调模式：get_sentence() 返回单个 dict；批量模式才返回 list
        if isinstance(sentence, dict):
            if RecognitionResult.is_sentence_end(sentence):
                self._parts.append(sentence.get("text", ""))
                self._partial = ""
            else:
                self._partial = sentence.get("text", "")
        else:
            for s in sentence:
                if not isinstance(s, dict):
                    continue
                if RecognitionResult.is_sentence_end(s):
                    self._parts.append(s.get("text", ""))
                    self._partial = ""
                else:
                    self._partial = s.get("text", "")

    @property
    def transcript(self) -> str:
        return "".join(self._parts) + self._partial

    def wait_closed(self, timeout: float = 8.0) -> None:
        self._closed.wait(timeout=timeout)


def create_streaming_recognizer() -> tuple[Recognition, _StreamCallback]:
    """创建并启动流式识别器，返回 (recognizer, callback)。"""
    cb = _StreamCallback()
    rec = Recognition(
        model=ASR_MODEL,
        format="pcm",
        sample_rate=16000,
        callback=cb,
        language_hints=["zh", "en"],
    )
    rec.start()
    return rec, cb


def finish_streaming_transcription(
    rec: Recognition,
    cb: _StreamCallback,
) -> tuple[str, float]:
    """
    停止流式识别并等待最终结果。

    Returns:
        (transcript, asr_post_stop_ms)  — asr_post_stop_ms 是 stop() 到拿到结果的耗时。
    """
    t0 = time.perf_counter()
    try:
        rec.stop()
    except Exception:
        pass  # 服务端可能已主动关闭（静音超时等），忽略
    cb.wait_closed()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if cb.error:
        # NO_VALID_AUDIO_ERROR = 录音太短或静音，视为空转录，不抛异常
        if "NO_VALID_AUDIO_ERROR" in cb.error:
            return "", elapsed_ms
        raise RuntimeError(f"DashScope streaming ASR error: {cb.error}")
    return cb.transcript.strip(), elapsed_ms


# ─── 批量 ASR（调试备用）──────────────────────────────────────────────────────

_batch_recognizer = Recognition(
    model=ASR_MODEL,
    format="wav",
    sample_rate=16000,
    callback=None,
    language_hints=["zh", "en"],
)


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> tuple[str, float]:
    """批量模式：录完后提交，返回 (transcript, elapsed_ms)。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        _write_wav(tmp, pcm_bytes, sample_rate)

    try:
        t0 = time.perf_counter()
        result = _batch_recognizer.call(tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        os.unlink(tmp_path)

    if result.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope ASR error {result.status_code}: {result.message}"
        )

    sentences = result.output.get("sentence", [])
    text = "".join(s.get("text", "") for s in sentences).strip()
    return text, elapsed_ms


def _write_wav(file_obj, pcm_bytes: bytes, sample_rate: int) -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    file_obj.write(buf.getvalue())


# ─── SDK 用：对预录 PCM 做流式转录 ───────────────────────────────────────────

def transcribe_pcm(pcm_bytes: bytes) -> tuple[str, float]:
    """
    对已录制完成的 PCM 音频做流式识别，返回 (transcript, elapsed_ms)。

    将音频按 30ms 帧逐块送入流式识别器，效果与实时流式相同，
    适用于 SDK 内部——调用方已完成录音，不需要边录边识别。

    Args:
        pcm_bytes: 16 kHz，16-bit signed，单声道 PCM 字节。

    Returns:
        (transcript, elapsed_ms)
    """
    rec, cb = create_streaming_recognizer()
    # 480 samples × 2 bytes = 960 bytes / chunk（30ms @ 16kHz）
    chunk_size = 960
    for i in range(0, len(pcm_bytes), chunk_size):
        rec.send_audio_frame(pcm_bytes[i: i + chunk_size])
    return finish_streaming_transcription(rec, cb)
