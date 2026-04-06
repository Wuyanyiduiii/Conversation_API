from __future__ import annotations

"""
main.py — 语音对话主入口。

在下方「模型配置」区将 None 改为字符串，可覆盖 .env 中对应的模型名称，
无需修改 .env 文件即可快速切换模型做对比测试。
"""

import json
import os
import warnings

# 屏蔽 macOS + LibreSSL 环境下 urllib3 v2 的 NotOpenSSLWarning
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

# ─── 模型配置（可在此覆盖 .env 中的模型名称）──────────────────────────────────
# 将 None 改为字符串即生效，例如:
#   ASR_MODEL_OVERRIDE = "whisper-large-v3"
#   LLM_MODEL_OVERRIDE = "deepseek-reasoner"
#   TTS_MODEL_OVERRIDE = "speech-01-hd"

ASR_MODEL_OVERRIDE: str | None = None
LLM_MODEL_OVERRIDE: str | None = None
TTS_MODEL_OVERRIDE: str | None = None
# ──────────────────────────────────────────────────────────────────────────────

# 场景数据文件（相对于本脚本所在目录）
SCENARIOS_FILE = "test.json"

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if ASR_MODEL_OVERRIDE:
    os.environ["ASR_MODEL"] = ASR_MODEL_OVERRIDE
if LLM_MODEL_OVERRIDE:
    os.environ["LLM_MODEL"] = LLM_MODEL_OVERRIDE
if TTS_MODEL_OVERRIDE:
    os.environ["TTS_MODEL"] = TTS_MODEL_OVERRIDE

# 以下 import 必须在 env override 之后，确保 config.py 读到最新值
from asr_client import create_streaming_recognizer, finish_streaming_transcription  # noqa: E402
from audio_io import AudioPlayer, record_with_streaming_asr  # noqa: E402
from config import ASR_MODEL, LLM_MODEL, TTS_MODEL, TTS_VOICE_ID  # noqa: E402
from conversation_sdk import ConversationSession  # noqa: E402


def _load_scenarios() -> list[dict]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, SCENARIOS_FILE)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_scenario(scenarios: list[dict]) -> dict:
    print("\n可用场景：")
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. 【{s['title']}】{s['subtitle']}")
    print()
    while True:
        raw = input("请输入场景编号（回车默认选 1）: ").strip()
        if raw == "":
            return scenarios[0]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(scenarios):
                return scenarios[idx]
        print(f"  请输入 1-{len(scenarios)} 之间的数字。")


def main() -> None:
    print("─" * 60)
    print(f"  ASR  : {ASR_MODEL}")
    print(f"  LLM  : {LLM_MODEL}")
    print(f"  TTS  : {TTS_MODEL}")
    print("─" * 60)

    scenarios = _load_scenarios()
    scenario = _select_scenario(scenarios)

    voice_id = scenario.get("tts_voice_id", TTS_VOICE_ID)
    print(f"\n▶ 场景：{scenario['title']} — {scenario['subtitle']}")
    print(f"  背景：{scenario['background']}")
    print(f"  你的任务：{scenario['task']}")
    print(f"  你的角色：{scenario['user_character']}")
    print(f"  音色：{voice_id}")
    print("\n按 Ctrl+C 退出\n")
    print("─" * 60)

    session = ConversationSession({
        "system_prompt": scenario["system_prompt"],
        "opening_line":  scenario.get("opening_line", ""),
        "tts_voice_id":  voice_id,
        "task":          scenario.get("task", ""),
    })
    player = AudioPlayer()

    # ── 播放 opening_line（AI 先开口）────────────────────────────────────────
    opening_text, opening_audio = session.get_opening()
    if opening_text:
        print(f"AI（开场）: {opening_text}")
        player.enqueue(opening_audio)

    try:
        while True:
            player.wait_done()

            # 流式 ASR：边录音边推流，stop 后极短时间拿到转录
            rec, cb = create_streaming_recognizer()
            try:
                _pcm, _t_stop = record_with_streaming_asr(rec.send_audio_frame)
                user_text, asr_ms = finish_streaming_transcription(rec, cb)
            except Exception as exc:
                print(f"\n[ASR 错误] {exc}\n")
                continue

            if not user_text.strip():
                continue

            print(f"\n你: {user_text}")

            # SDK 处理：LLM + 并发 TTS；on_audio_chunk 实现流式播放
            try:
                result = session.process_text_turn(
                    user_text,
                    asr_ms=asr_ms,
                    on_audio_chunk=player.enqueue,
                )
            except Exception as exc:
                print(f"\n[错误] {exc}\n")
                continue

            print(f"AI: {result.assistant_text or '……'}")
            print(f"\n[耗时] {result.timing.report()}\n")

            if result.end_flag == 1:
                player.wait_done()
                print("─" * 60)
                print("【对话已结束】")
                break

    except KeyboardInterrupt:
        print("\n退出。")
        player.wait_done()


if __name__ == "__main__":
    main()
