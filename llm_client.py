from __future__ import annotations

"""
llm_client.py — DeepSeek V3 对话封装。

LLM 输出严格约束为 JSON：{"text": "口语回复", "end": 0}
- text: 去除括号神态描写后的纯口语内容
- end:  0=继续对话，1=对话结束（耐心值耗尽或任务完成）
"""

import json
import re
import time

from openai import OpenAI

from config import DEEPSEEK_API_KEY, LLM_MODEL, MAX_HISTORY_TURNS

_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# ── 文本处理 ───────────────────────────────────────────────────────────────────
# 括号动作描写：匹配中文括号（…）或英文括号(…)，内容不超过 80 字
_BRACKET_RE = re.compile(r"[（(][^）)]{0,80}[）)]")
# 从 LLM 输出中提取 JSON 对象
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# 纯标点/省略号/空白——无有效语音内容，TTS 会播静音
_SILENT_ONLY_RE = re.compile(r"^[…。！？，、；：\s]+$")
# 流式解析：定位 text 字段值的开始位置（"text"\s*:\s*"）
_TEXT_INTRO_RE = re.compile(r'"text"\s*:\s*"')

# ── 句子切分 ───────────────────────────────────────────────────────────────────
_HARD_END = re.compile(r"[。！？…]+")
_SOFT_END = re.compile(r"[，、；]+")
_MIN_CHARS = 4
_MAX_BUFFER = 80


def split_sentences(text: str) -> list[str]:
    """将文本按句末标点切分为适合 TTS 的短句列表。"""
    sentences: list[str] = []
    buf = text
    while buf:
        m = _HARD_END.search(buf)
        if m and m.end() >= _MIN_CHARS:
            sentences.append(buf[: m.end()].strip())
            buf = buf[m.end() :]
        elif len(buf) >= _MAX_BUFFER:
            m2 = _SOFT_END.search(buf, _MAX_BUFFER // 2)
            if m2:
                sentences.append(buf[: m2.end()].strip())
                buf = buf[m2.end() :]
            else:
                sentences.append(buf[:_MAX_BUFFER].strip())
                buf = buf[_MAX_BUFFER :]
        else:
            if buf.strip():
                sentences.append(buf.strip())
            break
    return [s for s in sentences if s]


def _build_system_prompt(scenario: dict) -> str:
    base = scenario["system_prompt"].strip()
    task = scenario.get("task", "")
    return (
        f"{base}\n\n"
        "━━ 输出格式（强制，不得违反）━━\n"
        "你只能输出一个合法的 JSON 对象，格式如下，不得在 JSON 之外输出任何文字：\n"
        '{"text": "你的口语回复，1-3句话", "end": 0}\n\n'
        "字段规则：\n"
        "- text: 纯口语内容，严禁包含括号及括号内的动作/神态描写（如\"（后退半步）\"）；"
        "text 字段严禁为空字符串，如角色沉默或无话可说，也必须输出可朗读的极简口语如'嗯。'或'哼。'，严禁输出纯省略号或纯标点。\n"
        "- end: 整数 0 或 1。满足以下任一条件时置为 1，否则保持 0：\n"
        "  1. 客户耐心值已 ≤ 2，即将或已经结束对话；\n"
        f"  2. 销售任务已完成（任务描述：{task}）。"
    )


def _parse_response(raw: str) -> tuple[str, int]:
    """从 LLM 输出中解析 text 和 end_flag，容错处理非法 JSON。"""
    text = raw
    end_flag = 0
    try:
        m = _JSON_RE.search(raw)
        if m:
            obj = json.loads(m.group())
            text = str(obj.get("text", raw))
            end_flag = int(obj.get("end", 0))
    except (json.JSONDecodeError, ValueError):
        pass
    # 无论解析是否成功，都清除括号动作描写
    text = _BRACKET_RE.sub("", text).strip()
    # 过滤纯标点（如 "……"）——TTS 播出来是静音，视作无效响应
    if text and _SILENT_ONLY_RE.match(text):
        print(f"\n  [警告] LLM 返回纯标点内容 {text!r}，原始: {raw!r}", flush=True)
        text = ""
    elif not text:
        print(f"\n  [警告] LLM 返回空文本（括号过滤后），原始内容: {raw!r}", flush=True)
    return text, end_flag


def complete_turn(
    user_text: str,
    history: list[dict],
    scenario: dict,
) -> tuple[list[str], int, float | None, float]:
    """
    调用 LLM，返回分句列表、end_flag、TTFT 和总耗时。

    Args:
        user_text: 用户本轮输入文字。
        history:   对话历史（偶数条，user/assistant 交替）。
        scenario:  test.json 中的场景对象。

    Returns:
        (sentences, end_flag, ttft_ms, total_ms)
        sentences:  按句切分后的文本列表，供并发 TTS 使用。
        end_flag:   0=对话继续，1=对话结束。
        ttft_ms:    首 token 延迟（ms），流式获取。
        total_ms:   LLM 完整响应耗时（ms）。
    """
    sys_prompt = _build_system_prompt(scenario)
    base_messages = [{"role": "system", "content": sys_prompt}]
    base_messages.extend(history[-(MAX_HISTORY_TURNS * 2) :])
    base_messages.append({"role": "user", "content": user_text})

    for attempt in range(2):
        if attempt == 0:
            messages = base_messages
        else:
            messages = base_messages[:-1] + [{
                "role": "user",
                "content": user_text + '\n[请立即用JSON格式回复：{"text":"你的口语回复","end":0}]',
            }]

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        raw = ""

        stream = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=256,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            raw += delta

        total_ms = (time.perf_counter() - t_start) * 1000

        if raw.strip():
            break  # 有内容，不重试
        print(f"\n  [警告] LLM 返回空内容（第 {attempt+1} 次），正在重试...", flush=True)
    else:
        # 2 次都为空，用兑底词
        print("\n  [兑底] LLM 两次均返回空，使用兑底响应", flush=True)
        return ["嘅。"], 0, ttft_ms, total_ms

    text, end_flag = _parse_response(raw)
    sentences = split_sentences(text) or ["嘅。"]
    return sentences, end_flag, ttft_ms, total_ms


def stream_turn(
    user_text: str,
    history: list[dict],
    scenario: dict,
    max_history: int = MAX_HISTORY_TURNS,
):
    """
    流式调用 LLM，逐句 yield，边生成边供 TTS 使用。

    Yields:
        {"type": "sentence", "text": str}  — 每完成一句 yield 一次
        {"type": "done", "end_flag": int, "ttft_ms": float|None, "total_ms": float}

    解析策略（只关心 text 字段）：
    1. 在 token 流中等待出现 "text":" 模式，进入文本区。
    2. 逐字符扫描：遇到括号（/( 进入丢弃模式，）/) 退出；
       遇到句末标点且够长则切句 yield；遇到未转义的 " 则结束文本区。
    3. end_flag 从完整 raw 最终解析，不受流式截断影响。
    """
    sys_prompt = _build_system_prompt(scenario)
    base_messages = [{"role": "system", "content": sys_prompt}]
    base_messages.extend(history[-(max_history * 2):])
    base_messages.append({"role": "user", "content": user_text})

    for attempt in range(2):
        # 第二次重试时，在 user 消息末尾追加格式提醒，拉出 JSON Mode 卡住状态
        if attempt == 0:
            messages = base_messages
        else:
            messages = base_messages[:-1] + [{
                "role": "user",
                "content": user_text + '\n[请立即用JSON格式回复：{"text":"你的口语回复","end":0}]',
            }]

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        raw = ""

        stream = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=256,
        )

        header_buf = ""      # 进入 text 字段前的 token 缓冲
        in_text = False       # 是否已进入 text 字段值
        text_done = False     # text 字段值是否已被 " 关闭
        bracket_depth = 0    # 括号嵌套深度（>0 时内容丢弃）
        sentence_buf = ""    # 当前句子片段
        prev_ch = ""         # 上一字符，用于判断 \" 转义
        yielded_any = False  # 是否已 yield 过至少一句

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            raw += delta

            if text_done:
                continue

            if not in_text:
                header_buf += delta
                m = _TEXT_INTRO_RE.search(header_buf)
                if m:
                    in_text = True
                    delta = header_buf[m.end():]  # 开头引号之后的剩余内容
                    header_buf = ""
                else:
                    continue

            # 逐字符处理 text 字段内容
            for ch in delta:
                # 括号内容：丢弃
                if bracket_depth > 0:
                    if ch in '）)':
                        bracket_depth -= 1
                    elif ch in '（(':
                        bracket_depth += 1
                    prev_ch = ch
                    continue

                if ch in '（(':
                    bracket_depth += 1
                    prev_ch = ch
                    continue

                # 未转义的 " = text 字段结束
                if ch == '"' and prev_ch != '\\':
                    text_done = True
                    if sentence_buf.strip() and not _SILENT_ONLY_RE.match(sentence_buf.strip()):
                        yield {"type": "sentence", "text": sentence_buf.strip()}
                        yielded_any = True
                    sentence_buf = ""
                    break

                # 反斜杠是转义前缀，不计入文本
                if ch == '\\':
                    prev_ch = ch
                    continue

                sentence_buf += ch
                prev_ch = ch

                # 句末标点且满足最小长度 → 切句 yield
                if ch in '。！？' and len(sentence_buf.strip()) >= _MIN_CHARS:
                    s = sentence_buf.strip()
                    sentence_buf = ""
                    if not _SILENT_ONLY_RE.match(s):
                        yield {"type": "sentence", "text": s}
                        yielded_any = True

        total_ms = (time.perf_counter() - t_start) * 1000

        if raw.strip():
            break
        print(f"\n  [警告] LLM 返回空内容（第 {attempt + 1} 次），正在重试... raw={raw!r}", flush=True)
    else:
        # 2 次都为空，用兜底词
        print("\n  [兜底] LLM 两次均返回空，使用兜底响应", flush=True)
        yield {"type": "sentence", "text": "嗯。", "is_fallback": True}
        yield {"type": "done", "end_flag": 0, "ttft_ms": ttft_ms, "total_ms": total_ms}
        return
    # 兜底1：流异常中断时 flush sentence_buf 剩余内容
    if not text_done and sentence_buf.strip() and not _SILENT_ONLY_RE.match(sentence_buf.strip()):
        yield {"type": "sentence", "text": sentence_buf.strip()}
        yielded_any = True

    # 兜底2：模型未按 JSON 格式输出（in_text 始终为 False），直接将 raw 当作回复文本
    if not yielded_any and not in_text and raw.strip():
        fallback_text = _BRACKET_RE.sub("", raw).strip()
        if fallback_text and not _SILENT_ONLY_RE.match(fallback_text):
            print(f"\n  [非JSON] 模型未按格式输出，直接使用 raw: {raw!r}", flush=True)
            for s in split_sentences(fallback_text):
                yield {"type": "sentence", "text": s}
                yielded_any = True

    # 兜底3：整轮一句都没有（如 LLM 输出纯省略号），用"嗯。"代替
    if not yielded_any:
        print(f"\n  [兜底] 本轮无有效文本输出，以'嗯。'代替。raw={raw!r}", flush=True)
        yield {"type": "sentence", "text": "嗯。", "is_fallback": True}

    # end_flag 始终从完整 raw 解析
    _, end_flag = _parse_response(raw)
    yield {"type": "done", "end_flag": end_flag, "ttft_ms": ttft_ms, "total_ms": total_ms}
