"""
demo.py — ConversationSession SDK 使用示例。

演示最简洁的 Push-to-Talk 语音对话程序。
用户只需修改下方 CONFIG，即可接入自己的对话场景。

运行:
    python3 demo.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

from dotenv import load_dotenv
load_dotenv()

from asr_client import create_streaming_recognizer, finish_streaming_transcription
from audio_io import AudioPlayer, record_with_streaming_asr
from conversation_sdk import ConversationSession

# ── 在此定义你的对话场景 ─────────────────────────────────────────────────────
CONFIG = {
    # 必填：AI 的角色设定（系统提示词）
    "system_prompt": (
        "你现在是一名热情开朗的咖啡店店员，名叫小美，22岁。"
        "使用轻松自然的口吻与顾客交流，帮助推荐适合的咖啡。\n\n"
        "━━ 输出格式（强制）━━\n"
        '只输出 JSON：{"text": "你的口语回复，1-2句话", "end": 0}\n'
        "当顾客完成点单、说再见，或明确表示不需要帮助时，end 置为 1。"
    ),
    # 必填：AI 开场白（会先合成播放）
    "opening_line": "您好，欢迎光临！今天想喝点什么？",
    # 必填：MiniMax 音色 ID
    "tts_voice_id": "female-shaonv",
    # 可选：任务描述（帮助 AI 判断何时结束对话）
    "task": "完成顾客点单",
}
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    session = ConversationSession(CONFIG)
    player = AudioPlayer()

    print("─" * 50)
    print("  灵鹦语音对话 Demo")
    print("  操作：按 [空格] 开始录音，再按 [空格] 停止")
    print("  退出：Ctrl+C")
    print("─" * 50)

    # ── 播放开场白 ──────────────────────────────────────────────────────────
    opening_text, opening_audio = session.get_opening()
    if opening_text:
        print(f"\nAI: {opening_text}")
        player.enqueue(opening_audio)
        player.wait_done()

    # ── 对话循环 ────────────────────────────────────────────────────────────
    try:
        while True:
            player.wait_done()

            # 1. 流式 ASR：边录音边推流，stop 后极短时间拿到转录
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

            # 2. 调用 SDK 处理（LLM + TTS，on_audio_chunk 实现逐句流式播放）
            result = session.process_text_turn(
                user_text,
                asr_ms=asr_ms,
                on_audio_chunk=player.enqueue,
            )

            print(f"AI: {result.assistant_text or '……'}")
            print(f"[延迟] {result.timing.report()}\n")

            if result.end_flag == 1:
                player.wait_done()
                print("─" * 50)
                print("【对话已结束】")
                break

    except KeyboardInterrupt:
        print("\n退出。")
        player.wait_done()


if __name__ == "__main__":
    main()
