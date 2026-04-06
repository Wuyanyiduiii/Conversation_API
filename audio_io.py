"""
audio_io.py — 按键触发录音（Push-to-Talk）+ 扬声器顺序播放。

录音格式: 16 kHz / 16-bit 有符号 PCM / 单声道
播放格式: 32 kHz / 16-bit 有符号 PCM / 单声道（MiniMax 输出格式）

操作方式: 按 [空格] 开始录音，再按 [空格] 停止并进入处理流程。
"""
from __future__ import annotations

import queue
import sys
import termios
import threading
import time
import tty
from typing import Callable

import numpy as np
import sounddevice as sd

from tts_client import TTS_SAMPLE_RATE

# ── 录音参数 ──────────────────────────────────────────────────────────────────
RECORD_SAMPLE_RATE: int = 16000
RECORD_CHANNELS: int = 1
FRAME_SAMPLES: int = 480  # 30ms @ 16kHz


# ── 按键读取 ──────────────────────────────────────────────────────────────────
def _read_one_char() -> str:
    """从 stdin 读取单个字符，不回显、不等待回车。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    return ch


def _wait_for_space() -> None:
    """阻塞直到用户按下空格键。"""
    while True:
        if _read_one_char() == " ":
            return


def record_push_to_talk() -> bytes:
    """
    按 [空格] 开始录音，再按 [空格] 停止。

    Returns:
        原始 16-bit PCM 字节（16 kHz，单声道）。
    """
    print("  按 [空格] 开始说话…", end="", flush=True)
    _wait_for_space()
    print("  录音中… 再按 [空格] 结束", end="", flush=True)

    frames: list[bytes] = []
    audio_q: queue.Queue[bytes] = queue.Queue()
    stop_event = threading.Event()

    def _callback(indata, frame_count, time_info, status) -> None:  # noqa: ARG001
        audio_q.put(bytes(indata))

    def _drain() -> None:
        while not stop_event.is_set():
            try:
                frames.append(audio_q.get(timeout=0.05))
            except queue.Empty:
                pass
        # 清空剩余缓冲
        while not audio_q.empty():
            try:
                frames.append(audio_q.get_nowait())
            except queue.Empty:
                break

    with sd.RawInputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=RECORD_CHANNELS,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=_callback,
    ):
        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        _wait_for_space()
        stop_event.set()
        drain_thread.join()

    print("  完毕")
    return b"".join(frames)


def record_with_streaming_asr(
    send_chunk_fn: Callable[[bytes], None],
) -> tuple[bytes, float]:
    """
    按 [空格] 开始录音，同时将每个音频块送入 send_chunk_fn（流式 ASR）；
    再按 [空格] 停止。

    Returns:
        (pcm_bytes, t_stop) — t_stop 是 perf_counter() 在 Stop 键被按下那一刻的值，
        用于后续精确计算端到端延迟。
    """
    print("  按 [空格] 开始说话…", end="", flush=True)
    _wait_for_space()
    print("  录音中… 再按 [空格] 结束", end="", flush=True)

    frames: list[bytes] = []
    audio_q: queue.Queue[bytes] = queue.Queue()
    stop_event = threading.Event()
    t_stop_holder: list[float] = [0.0]

    def _callback(indata, frame_count, time_info, status) -> None:  # noqa: ARG001
        audio_q.put(bytes(indata))

    def _drain_and_stream() -> None:
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.02)
                frames.append(chunk)
                try:
                    send_chunk_fn(chunk)
                except Exception:
                    return  # ASR 已被服务端关闭，不再推流
            except queue.Empty:
                pass
        # 排空剩余缓冲
        while not audio_q.empty():
            try:
                chunk = audio_q.get_nowait()
                frames.append(chunk)
                try:
                    send_chunk_fn(chunk)
                except Exception:
                    break  # ASR 已停止，跳出
            except queue.Empty:
                break

    with sd.RawInputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=RECORD_CHANNELS,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=_callback,
    ):
        drain_thread = threading.Thread(target=_drain_and_stream, daemon=True)
        drain_thread.start()
        _wait_for_space()
        t_stop_holder[0] = time.perf_counter()  # 精确在按键瞬间打时间戳
        stop_event.set()
        drain_thread.join()

    print("  完毕")
    return b"".join(frames), t_stop_holder[0]


class AudioPlayer:
    """
    线程安全的顺序音频播放器。

    调用 enqueue() 将 PCM 块加入队列，后台线程按顺序播放，
    保证多句 TTS 结果严格按原始句子顺序输出。
    """

    def __init__(self) -> None:
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, pcm_bytes: bytes) -> None:
        """将 32 kHz / 16-bit / 单声道 PCM 加入播放队列。"""
        self._q.put(pcm_bytes)

    def wait_done(self) -> None:
        """阻塞直到队列中所有音频播放完毕。"""
        self._q.join()

    def _worker(self) -> None:
        while True:
            pcm = self._q.get()
            if pcm is None:
                self._q.task_done()
                break
            try:
                arr = np.frombuffer(pcm, dtype=np.int16)
                sd.play(arr, samplerate=TTS_SAMPLE_RATE)
                sd.wait()
            finally:
                self._q.task_done()
