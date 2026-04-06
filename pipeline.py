"""
pipeline.py — 录音（流式ASR）→ LLM（JSON 模式）→ TTS（并发）→ 顺序播放。

录音与 ASR 同步进行，Stop 键按下后只需等 ASR 收尾（~100-300ms），
大幅降低端到端延迟。
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor

from asr_client import create_streaming_recognizer, finish_streaming_transcription
from audio_io import AudioPlayer, record_with_streaming_asr
from llm_client import split_sentences, stream_turn
from timing import TurnTiming
from tts_client import synthesize


def _run_tts_concurrent(
    sentences: list[str],
    player: AudioPlayer,
    timing: TurnTiming,
    t_total: float,
) -> None:
    """对 sentences 并发合成 TTS，按原句序入队播放。"""
    if not sentences:
        return

    tts_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts")
    futures: list[Future] = [tts_pool.submit(synthesize, s) for s in sentences]
    tts_pool.shutdown(wait=False)

    first = True
    for fut in futures:
        pcm, tts_ms = fut.result()
        if first:
            timing.tts_first_chunk_ms = tts_ms
            timing.total_ms = (time.perf_counter() - t_total) * 1000
            first = False
        player.enqueue(pcm)

    if timing.total_ms is None:
        timing.total_ms = (time.perf_counter() - t_total) * 1000


def speak_opening_line(opening_line: str, player: AudioPlayer) -> None:
    """将 opening_line 合成并播放，作为 AI 的开场白。"""
    sentences = split_sentences(opening_line)
    if not sentences:
        return
    dummy_timing = TurnTiming()
    _run_tts_concurrent(sentences, player, dummy_timing, time.perf_counter())
    player.wait_done()


def run_turn(
    history: list[dict],
    player: AudioPlayer,
    scenario: dict,
) -> tuple[str, str, TurnTiming, int]:
    """
    完整的一轮对话：录音（流式ASR同步推流）→ LLM（JSON）→ 并发 TTS → 入队播放。

    Args:
        history:   对话历史列表，原地更新。
        player:    共享的 AudioPlayer 实例。
        scenario:  test.json 中的当前场景对象。

    Returns:
        (user_text, assistant_text, TurnTiming, end_flag)
        end_flag: 0=继续, 1=对话结束
    """
    timing = TurnTiming()

    # ── 流式 ASR + 录音（同步进行）────────────────────────────────────────────
    rec, cb = create_streaming_recognizer()
    _pcm_bytes, t_stop = record_with_streaming_asr(rec.send_audio_frame)

    # t_stop = Space 键被按下的精确时刻，作为端到端延迟的起点
    user_text, timing.asr_ms = finish_streaming_transcription(rec, cb)

    if not user_text.strip():
        return "", "", timing, 0

    print(f"\n你: {user_text}")

    # ── 流式 LLM → 每句立即提交 TTS（与后续 LLM 生成并行）─────────────────
    tts_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts")
    futures: list[Future] = []
    collected: list[str] = []
    is_fallback = False
    end_flag = 0

    for event in stream_turn(user_text, history, scenario):
        if event["type"] == "sentence":
            s = event["text"]
            collected.append(s)
            futures.append(tts_pool.submit(synthesize, s))
            if event.get("is_fallback"):
                is_fallback = True
        elif event["type"] == "done":
            timing.llm_ttft_ms = event["ttft_ms"]
            timing.llm_total_ms = event["total_ms"]
            end_flag = event["end_flag"]

    tts_pool.shutdown(wait=False)

    assistant_text = "".join(collected)
    print(f"AI: {assistant_text or '……'}")

    if not futures:
        # LLM 完全没有输出（极罕见），补一条占位 assistant 保持 user/assistant 交替结构
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": "嗯。"})
        timing.total_ms = (time.perf_counter() - t_stop) * 1000
        return user_text, "", timing, end_flag

    # 按原句序收集 PCM 并入队（TTS 已在 LLM 生成期间并发完成大部分工作）
    first = True
    for fut in futures:
        pcm, tts_ms = fut.result()
        if first:
            timing.tts_first_chunk_ms = tts_ms
            timing.total_ms = (time.perf_counter() - t_stop) * 1000
            first = False
        player.enqueue(pcm)

    # 兜底内容（is_fallback）写入简短占位，保持 user/assistant 结构，但不暴露具体内容
    if not is_fallback:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
    else:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": "嗯。"})

    return user_text, assistant_text, timing, end_flag
