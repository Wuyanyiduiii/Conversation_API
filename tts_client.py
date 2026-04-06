"""
tts_client.py — MiniMax TTS 封装。

将文字合成为 32 kHz / 16-bit 有符号 PCM 音频字节。
"""
from __future__ import annotations

import time

import httpx

from config import MINIMAX_API_KEY, MINIMAX_GROUP_ID, TTS_MODEL, TTS_VOICE_ID

_TTS_ENDPOINT = "https://api.minimax.chat/v1/t2a_v2"

# 音频格式常量（与 audio_io.py 保持一致）
TTS_SAMPLE_RATE: int = 32000  # Hz
TTS_SAMPLE_WIDTH: int = 2     # bytes，16-bit


def synthesize(text: str, voice_id: str | None = None) -> tuple[bytes, float]:
    """
    将文字合成为原始 PCM 音频。

    Args:
        text:     待合成的文字。
        voice_id: MiniMax 音色 ID；为 None 时使用 .env 中的 TTS_VOICE_ID。

    Returns:
        (pcm_bytes, elapsed_ms)
        pcm_bytes: 32 kHz，16-bit 有符号，单声道。
    """
    _voice_id = voice_id or TTS_VOICE_ID
    payload = {
        "model": TTS_MODEL,
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": _voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": TTS_SAMPLE_RATE,
            "bitrate": 128000,
            "format": "pcm",
            "channel": 1,
        },
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            _TTS_ENDPOINT,
            params={"GroupId": MINIMAX_GROUP_ID},
            headers=headers,
            json=payload,
        )
    resp.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    body = resp.json()
    status_code = body.get("base_resp", {}).get("status_code", -1)
    if status_code != 0:
        status_msg = body.get("base_resp", {}).get("status_msg", "unknown")
        raise RuntimeError(f"MiniMax TTS error {status_code}: {status_msg}")

    audio_hex: str = body["data"]["audio"]
    pcm_bytes = bytes.fromhex(audio_hex)
    return pcm_bytes, elapsed_ms
