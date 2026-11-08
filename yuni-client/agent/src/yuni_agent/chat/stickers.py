"""Read existing library assets and select companion stickers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select

from yuni_agent.storage.models import ChatMessage, Sticker

logger = logging.getLogger(__name__)

STICKER_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
STICKER_SELECT_TIMEOUT = 10.0

EXPLICIT_USER_STICKER_RE = re.compile(
    r"(?:换|发|来|看|挑|给)(?:一?张|个)?表情|(?:那你|你再)?(?:换|发|挑|来)一?张|看看你能(?:换|发)什么|斗图"
)
EXPLICIT_REPLY_STICKER_RE = re.compile(
    r"这张呢|看这张|换了?这张|挑张像样的|发张表情|给你看张"
)

EMOTION_SYNONYMS: dict[str, list[str]] = {
    "开心": ["开心", "高兴", "愉快", "快乐", "喜悦", "兴奋", "哈哈", "乐", "俏皮", "活泼", "欢快"],
    "喜欢": ["喜欢", "亲昵", "宠溺", "心动", "温柔", "甜蜜", "贴贴", "爱", "体贴", "暖心"],
    "撒娇": ["撒娇", "黏人", "求关注", "讨好", "软萌", "哄", "卖萌", "求理睬", "抱抱"],
    "害羞": ["害羞", "羞涩", "不好意思", "脸红", "腼腆"],
    "得意": ["得意", "傲娇", "自豪", "炫耀", "臭屁", "神气", "嘴硬", "耍帅"],
    "期待": ["期待", "好奇", "盼望", "等着", "星星眼", "感兴趣", "催促"],
    "感谢": ["感谢", "谢谢", "感激"],
    "安慰": ["安慰", "心疼", "抱抱", "摸摸头", "关心", "安抚", "体谅"],
    "加油": ["加油", "鼓励", "打气", "支持", "冲"],
    "赞同": ["赞同", "认可", "同意", "夸奖", "点赞", "收到", "明白", "配合", "好主意"],
    "调侃": ["调侃", "打趣", "逗", "玩笑", "吐槽", "戏谑", "坏笑", "揶揄", "欠揍", "搞笑", "幽默"],
    "敷衍": ["敷衍", "随口", "漫不经心", "哦"],
    "委屈": ["委屈", "小委屈", "求原谅", "可怜", "哭唧唧", "被晾着", "幽怨"],
    "难过": ["难过", "伤心", "失落", "低落", "哭", "心酸"],
    "无奈": ["无奈", "叹气", "哭笑不得", "拿你没办法", "服了", "妥协", "头疼", "心累"],
    "生气": ["生气", "恼火", "不满", "抗议", "炸毛", "凶", "吃醋", "记仇"],
    "无语": ["无语", "汗颜", "沉默", "呆住", "无言以对"],
    "嫌弃": ["嫌弃", "鄙视", "白眼", "不屑", "吐槽"],
    "震惊": ["震惊", "惊讶", "意外", "吃惊", "目瞪口呆", "吃瓜"],
    "疑惑": ["疑惑", "好奇", "不解", "问号", "困惑", "懵", "纳闷"],
    "思考": ["思考", "琢磨", "认真", "思索"],
    "困倦": ["困倦", "犯困", "疲惫", "累", "懒洋洋", "慵懒", "打哈欠", "想睡", "熬夜"],
    "崩溃": ["崩溃", "抓狂", "破防", "头大", "受不了"],
    "摆烂": ["摆烂", "躺平", "摸鱼", "偷懒", "放空", "悠闲", "歇会儿", "不想动"],
    "道歉": ["道歉", "认错", "对不起", "心虚"],
    "打招呼": ["打招呼", "问候", "早安", "嗨", "冒泡", "戳一下", "在干嘛"],
    "晚安": ["晚安", "睡觉", "好梦", "早点睡", "休息"],
}


def sticker_path(data_dir: Path, sticker: Sticker) -> Path:
    root = (data_dir / "stickers").resolve()
    target = (data_dir / sticker.relative_path).resolve()
    if not root.is_relative_to(data_dir.resolve()) or not target.is_relative_to(root) or sticker.content_type not in STICKER_TYPES:
        raise ValueError("表情素材无效")
    return target


def sticker_data(sticker_id: int, description: str) -> dict:
    return {"id": sticker_id, "description": description, "imageUrl": f"/api/chat/stickers/{sticker_id}/content"}


def _tokens(value: str) -> list[str]:
    values: list[str] = []
    for part in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", value.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            values.extend(part[i:i + 2] for i in range(max(1, len(part) - 1)))
        else:
            values.append(part)
    return values


def _matched_emotions(text: str) -> set[str]:
    matched: set[str] = set()
    for emotion, keywords in EMOTION_SYNONYMS.items():
        if any(kw in text for kw in keywords):
            matched.add(emotion)
    return matched


class StickerSelector:
    """Recall and select 0 or 1 companion sticker for an assistant reply."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def _recall_candidates(
        self,
        items: list[dict[str, Any]],
        *,
        plan: Any,
        reply_text: str,
        user_text: str,
        inline_hint: str = "",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        intent_dump = plan.expression_intent.model_dump()
        intent_text = " ".join(
            "、".join(v) if isinstance(v, list) else str(v)
            for k, v in intent_dump.items()
            if k != "avoid" and v
        )
        query = f"{intent_text} {plan.reply_reference} {reply_text} {inline_hint} {user_text}".strip()
        target_emotions = _matched_emotions(f"{intent_text} {reply_text} {inline_hint}")

        docs = {
            item["id"]: Counter(_tokens(f"{' '.join(item['emotions'])} {item['usage']} {item['description']}"))
            for item in items
        }
        q_terms = set(_tokens(query))
        avg_len = sum(sum(d.values()) for d in docs.values()) / max(1, len(docs)) or 1.0
        df = {term: sum(term in d for d in docs.values()) for term in q_terms}

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            doc = docs[item["id"]]
            doc_len = sum(doc.values()) or 1
            bm25 = 0.0
            for term in q_terms:
                tf = doc.get(term, 0)
                if tf:
                    idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                    bm25 += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * doc_len / avg_len))
            emotion_hits = sum(
                (2.5 if idx == 0 else 1.5)
                for idx, emo in enumerate(item["emotions"])
                if emo in target_emotions
            )
            score = emotion_hits + bm25 * 0.35 + random.uniform(0, 0.15)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Keep diversity across primary emotions so the model is not handed 8 near-identical items.
        selected: list[dict[str, Any]] = []
        per_emotion: Counter[str] = Counter()
        for score, item in scored:
            if len(selected) >= 3 and score <= 0.25 and len(scored) > limit:
                break
            primary = item["emotions"][0] if item["emotions"] else "其他"
            if per_emotion[primary] >= 3 and len(scored) > limit:
                continue
            selected.append(item)
            per_emotion[primary] += 1
            if len(selected) >= limit:
                break
        return selected or [pair[1] for pair in scored[: min(limit, len(scored))]]

    async def select(
        self,
        state: dict[str, Any],
        plan: Any,
        reply_text: str,
        client: Any,
        *,
        inline_hint: str = "",
    ) -> dict[str, Any] | None:
        # At most one sticker per turn (only on the first reply of the turn); a reply
        # carrying a generated image never gets a sticker as well.
        if state.get("reply_count", 0) > 0 or state.get("pending_image"):
            return None
        user_text = str(state.get("current_input") or "")
        explicit_sticker_intent = bool(
            inline_hint.strip()
            or EXPLICIT_USER_STICKER_RE.search(user_text)
            or EXPLICIT_REPLY_STICKER_RE.search(reply_text)
        )
        # Serious / long explanatory replies or tool execution turns rarely need a meme sticker.
        if not explicit_sticker_intent and (plan.reply_style == "展开" or state.get("tool_results")):
            return None

        generation = self.runtime.data_generation
        conversation_id = state.get("conversation_id")
        user_message_id = state.get("user_message_id")
        current_assistant_id = state.get("assistant_message_id")
        data_dir = self.runtime.config.data_dir

        with self.runtime.require_database().session() as session:
            user_msg = session.get(ChatMessage, user_message_id) if user_message_id else None
            user_sent_sticker = bool(user_msg and user_msg.sticker)
            user_sticker_id = (user_msg.sticker or {}).get("id") if user_msg and user_msg.sticker else None

            conditions = [
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.role == "assistant",
                ChatMessage.status == "succeeded",
            ]
            if current_assistant_id is not None:
                conditions.append(ChatMessage.id != current_assistant_id)
            recent_assistants = list(
                session.scalars(
                    select(ChatMessage)
                    .where(*conditions)
                    .order_by(ChatMessage.id.desc())
                    .limit(6)
                )
            )
            # Rhythm guard: if the assistant just sent a sticker in the last 2 assistant messages,
            # skip unless the user also sent a sticker in this turn ("斗图/回敬表情包")
            # or the conversation explicitly asks/promises to send a sticker.
            assistant_recent_sticker = any(row.sticker for row in recent_assistants[:2])
            if assistant_recent_sticker and not user_sent_sticker and not explicit_sticker_intent:
                return None

            recent_sticker_ids = {
                (row.sticker or {}).get("id")
                for row in recent_assistants
                if isinstance(row.sticker, dict) and (row.sticker or {}).get("id") is not None
            }
            if user_sticker_id is not None:
                recent_sticker_ids.add(user_sticker_id)

            rows = list(session.scalars(select(Sticker).where(Sticker.enabled.is_(True)).order_by(Sticker.id)))
            pool: list[dict[str, Any]] = []
            fallback_pool: list[dict[str, Any]] = []
            for row in rows:
                try:
                    if not sticker_path(data_dir, row).is_file():
                        continue
                except ValueError:
                    continue
                entry = {
                    "id": row.id,
                    "description": row.description,
                    "emotions": list(row.emotions or []),
                    "usage": row.usage or "",
                }
                fallback_pool.append(entry)
                if row.id not in recent_sticker_ids:
                    pool.append(entry)
            if not pool:
                pool = fallback_pool

        if not pool:
            return None

        started = time.perf_counter()
        candidates = self._recall_candidates(
            pool, plan=plan, reply_text=reply_text, user_text=user_text, inline_hint=inline_hint, limit=8
        )
        if not candidates:
            return None

        candidate_view = [
            {
                "id": item["id"],
                "emotions": item["emotions"],
                "usage": item["usage"],
                "description": item["description"],
            }
            for item in candidates
        ]
        intent_view = {
            k: v for k, v in plan.expression_intent.model_dump().items()
            if (v if isinstance(v, list) else str(v).strip())
        }
        rule_3 = (
            "3. 当前对话中用户明确要求伴侣发/换表情包，或伴侣台词已明确表示正在发出一张表情包，因此本次【必须】从候选中选出最贴合伴侣台词语气的一张并返回其整数 sticker_id，不得返回 null。\n"
            if explicit_sticker_intent
            else (
                "3. 用户本轮发了表情包，若候选中有恰当的互动表情包可优先考虑回敬一张；若都不贴切仍返回 null。\n"
                if user_sent_sticker
                else "3. 用户本轮未发表情包，请克制选择，仅在情绪表现力明显（如撒娇、打趣、害羞、委屈、欢快、困倦道晚安等）且候选极度贴合时才选，普通平淡对话返回 null。\n"
            )
        )
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "你负责判断虚拟伴侣（assistant）在本轮文字回复后，是否顺手附带一张表情包来传递情绪或活跃气氛。\n"
                    "选择原则：\n"
                    "1. 表情包是点缀而非必选项：只有当某张候选表情包的情绪（emotions）、使用场景（usage）和画面与伴侣本轮台词的语气高度契合、发出来自然生动时才选；若台词本身已足够、话题严肃，或没有特别贴切的候选，必须返回 null。\n"
                    "2. 注意区分主客体立场：选出的表情包代表【伴侣自己】此刻对用户说话时的神态、心情或互动动作（如撒娇、得意、害羞、委屈、调侃、打招呼、道晚安），不要选成与伴侣台词立场相反的图。\n"
                    + rule_3
                    + '严格只输出 JSON 对象：{"sticker_id": 整数ID或null}。'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": user_text[:300],
                        "assistant_reply": reply_text,
                        **({"sticker_hint": inline_hint[:200]} if inline_hint else {}),
                        "expression_intent": intent_view,
                        "candidates": candidate_view,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        result = None
        try:
            async with asyncio.timeout(STICKER_SELECT_TIMEOUT):
                internal_client = client.for_internal_json() if hasattr(client, "for_internal_json") else client
                result = await internal_client.complete_json(prompt_messages)
        except Exception as exc:
            logger.warning("[Sticker Select] model call failed (%s); skipping sticker", type(exc).__name__)
            if not explicit_sticker_intent:
                return None

        if generation != self.runtime.data_generation:
            return None

        allowed = {item["id"]: item for item in candidates}
        chosen_id = result.get("sticker_id") if isinstance(result, dict) and set(result) == {"sticker_id"} else None
        if chosen_id is None or type(chosen_id) is not int or chosen_id not in allowed:
            if explicit_sticker_intent and candidates:
                chosen = candidates[0]
                logger.info(
                    "[Sticker Select] fallback_explicit=%s emotions=%s description=%s elapsed_ms=%.1f",
                    chosen["id"], chosen["emotions"], chosen["description"], (time.perf_counter() - started) * 1000,
                )
                return {
                    "id": chosen["id"],
                    "description": chosen["description"],
                    "emotions": chosen["emotions"],
                }
            logger.info("[Sticker Select] selected=None elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
            return None

        chosen = allowed[chosen_id]
        logger.info(
            "[Sticker Select] selected=%s emotions=%s description=%s elapsed_ms=%.1f",
            chosen["id"], chosen["emotions"], chosen["description"], (time.perf_counter() - started) * 1000,
        )
        return {
            "id": chosen["id"],
            "description": chosen["description"],
            "emotions": chosen["emotions"],
        }

