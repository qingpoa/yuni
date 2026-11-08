from __future__ import annotations

import random

from yuni_agent.model.client import ModelClientError

REPLY_OUTPUT_INSTRUCTIONS = "给出日常且口语化的纯文本回复，只输出角色说出口的台词，不使用 JSON、代码围栏、旁白或动作注释。"

def parse_reply(content: str) -> list[str]:
    """Adapt MaiBot's protected segmentation/adjacent grouping without rewriting text."""
    text = content.strip()
    if not text or len(text) > 8000:
        raise ModelClientError("模型回复为空或过长，请重试")
    pairs = {'“': '”', '‘': '’', '「': '」', '『': '』', '（': '）', '(': ')', '[': ']', '【': '】', '《': '》', '"': '"', "'": "'"}
    stack: list[str] = []
    boundaries: list[int] = []
    start = 0
    fenced = False
    for i, char in enumerate(text):
        if text[i:i + 3] == '```':
            fenced = not fenced
        if fenced:
            continue
        if char == "'" and i and i + 1 < len(text) and text[i - 1].isalnum() and text[i + 1].isalnum():
            continue
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        if stack:
            continue
        split = char in '。！？!?；;\n'
        if split:
            # Keep separator runs and whitespace with the preceding segment.
            if i + 1 < len(text) and (text[i + 1] in '。！？!?；;，,\n ' or text[i + 1].isspace()):
                continue
            boundaries.append(i + 1)
            start = i + 1
    segments = []
    start = 0
    for end in [*boundaries, len(text)]:
        if end > start:
            segments.append(text[start:end])
        start = end
    if len(text) < 20:
        split_strength = 0.2
    elif len(text) < 42:
        split_strength = 0.45
    else:
        split_strength = 0.65
    merge_probability = 1.0 - split_strength
    merged: list[str] = []
    idx = 0
    while idx < len(segments):
        current = segments[idx]
        if idx + 1 < len(segments) and not current.endswith('\n') and random.random() < merge_probability:
            merged.append(current + segments[idx + 1])
            idx += 2
        else:
            merged.append(current)
            idx += 1
    segments = merged
    if len(text) < 12:
        return [text]
    # Same adjacent, evenly-sized grouping as MaiBot; no dropped separators.
    parts = []
    start = 0
    for group in range(min(3, len(segments))):
        count = (len(segments) - start + (min(3, len(segments)) - group) - 1) // (min(3, len(segments)) - group)
        parts.append(''.join(segments[start:start + count]))
        start += count
    return parts
