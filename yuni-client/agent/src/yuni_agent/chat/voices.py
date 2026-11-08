"""Select a voice reply once, clean its transcript and synthesize safely."""
from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any

from sqlalchemy import select

from yuni_agent.model.voice import VOICE_TAG_RE, store_voice, synthesize_speech
from yuni_agent.storage.models import ChatMessage

logger = logging.getLogger(__name__)
VOICE_SYNTHESIS_TIMEOUT = 25.0
# Shared grammar keeps requests and their negations aligned across quantities.
VOICE_FILLER = r"(?:一下|下|给我|向我|对我|替我|帮我|跟|和|我|你|再|先|多|少|点|个|条|段|句|则|份|声|次|小|短|[一二两三四五六七八九十几\d]|\s)"
VOICE_SEND_ACTION = rf"(?:发(?:送|出)?|来|录(?:制)?|说|讲|回(?:复)?|念|读|唱|整|搞|弄|甩|传|给){VOICE_FILLER}{{0,6}}语音"
VOICE_FOLLOWUP_RE = re.compile(
    r"^(?:那就|那|请|快|再|在)?\s*(?:再|在)(?:来|说|讲|发|录|念|唱)(?:一?[句条段个遍声次]|两句|几句)"
    r"|没听清|听不清|大声点|小声点|多说(?:两|几)?句|声音(?:太小|太大|真好听|好听)"
)
PAST_OR_META_VOICE_RE = re.compile(
    r"(?:刚才|刚刚|上次|之前|上一[条句段个])[^，。！？?\n]{0,8}语音[^，。！？?\n]{0,6}(?:了吗|了没|吗|呢|是什么|说了什么)"
    r"|语音(?:功能|设置|开关|配置|模式|权限)[^，。！？?\n]{0,8}(?:怎么|如何|在哪|是什么)"
)
EXPLICIT_USER_VOICE_RE = re.compile(
    VOICE_SEND_ACTION
    + r"|听(?:一?听)?(?:你(?:的)?|点)?(?:声音|语音|说话|亲口说)|想听(?:你(?:的)?)?(?:声音|语音|说话|亲口说)"
    + r"|(?:用|以|拿|靠)语音|语音(?:说|回(?:复|应)?|聊|讲|念|读|唱|哄|发|来|告诉|跟我|和我|给我|向我|对我|吧|呗|啊|呀|嘛|哦|喔|呢|啦)"
    + r"|(?:念|读|唱|说)给我听|(?:亲口|开口)(?:跟我|和我|对我|给我)?(?:说|讲|念)"
    + r"|^(?:那就|那|请|快|先)?\s*语音[啊呀嘛吧哦喔呢呗啦！!。？?\s]*$"
)
NEGATIVE_VOICE_RE = re.compile(
    rf"(?:别|不要|不用|不想|无需|不必|甭|莫)(?:再|发(?:送|出)?|来|录(?:制)?|说|讲|回(?:复)?|念|读|唱|整|搞|弄|甩|传|给|听|用|{VOICE_FILLER})*语音"
    r"|发文字|用文字|打字就好|只要文字|文字回复|回文字"
)
PLAN_VOICE_INTENT_RE = re.compile(
    r"(?:发(?:送)?|用|以|通过|回(?:复|应)?|来|录)[^，。；\n]{0,6}语音|语音(?:回(?:复|应)?|消息|形式|说|表达)"
)
PLAN_VOICE_NEGATIVE_RE = re.compile(r"(?:不|勿|别|无需|不用|无法|不能)(?:发|用|使用)?语音|文字回复|纯文字")
KAOMOJI_RE = re.compile(r"QAQ|QwQ|T_T|>_<|\^_\^", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\ufe0e\ufe0f\u200d\u20e3]"
)


def clean_text_for_tts(text: str) -> str:
    text = VOICE_TAG_RE.sub("", text)
    text = re.sub(r"\([^()]*\)|（[^（）]*）|\*[^*\n]*\*", "", text)
    text = KAOMOJI_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    # Preserve Chinese punctuation: it controls natural pauses in synthesis.
    return " ".join(text.split()).strip()


class VoiceSelector:
    def __init__(self, runtime: Any, *, rng: random.Random | None = None) -> None:
        self.runtime = runtime
        self._rng = rng or random.Random()

    @staticmethod
    def _has_explicit_voice_intent(user_text: str, plan: Any) -> bool:
        if NEGATIVE_VOICE_RE.search(user_text) or PAST_OR_META_VOICE_RE.search(user_text):
            return False
        if EXPLICIT_USER_VOICE_RE.search(user_text):
            return True
        if not any(token in user_text for token in ("语音", "声音", "念给我听", "亲口", "开口")):
            return False
        intent = plan.expression_intent.model_dump() if hasattr(plan, "expression_intent") else {}
        plan_text = " ".join(
            str(part)
            for part in (
                getattr(plan, "reply_reference", ""),
                intent.get("reply_act", ""),
                intent.get("scene", ""),
                " ".join(intent.get("prefer", []) or []),
            )
            if part
        )
        return bool(PLAN_VOICE_INTENT_RE.search(plan_text) and not PLAN_VOICE_NEGATIVE_RE.search(plan_text))

    def should_reply_with_voice(self, state: dict[str, Any], plan: Any) -> bool:
        if (
            not self.runtime.voice_generation_ready
            or state.get("reply_count", 0) > 0
            or state.get("pending_image")
        ):
            return False
        user_text = str(state.get("current_input") or "")
        if NEGATIVE_VOICE_RE.search(user_text) or PAST_OR_META_VOICE_RE.search(user_text):
            return False
        if self._has_explicit_voice_intent(user_text, plan):
            return True
        if plan.reply_style == "展开" or state.get("tool_results"):
            return False
        if getattr(plan, "use_voice", False):
            return True
        with self.runtime.require_database().session() as session:
            recent = list(session.scalars(select(ChatMessage).where(
                ChatMessage.conversation_id == state["conversation_id"],
                ChatMessage.role == "assistant", ChatMessage.status == "succeeded",
            ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(2)))
        if recent and recent[0].voice and VOICE_FOLLOWUP_RE.search(user_text):
            return True
        auth = self.runtime.voice_auth
        if auth.trigger_mode == "explicit_only":
            return False
        if auth.trigger_mode == "always":
            return True
        if any(message.voice for message in recent):
            return False
        intent = plan.expression_intent.model_dump()
        scene = f"{intent.get('tone', '')} {intent.get('scene', '')} {intent.get('reply_act', '')}"
        emotional = any(word in scene for word in ("亲昵", "撒娇", "早安", "晚安", "安慰", "打招呼"))
        return self._rng.random() < (0.30 if emotional else 0.12)

    async def synthesize_for_reply(
        self, state: dict[str, Any], reply_text: str,
    ) -> dict[str, Any] | None:
        if not self.runtime.voice_generation_ready:
            return None
        cleaned = clean_text_for_tts(reply_text)
        if not cleaned or len(cleaned) > 120 or "```" in reply_text:
            return None
        auth = self.runtime.voice_auth
        generation = self.runtime.data_generation
        try:
            audio_bytes = await asyncio.wait_for(synthesize_speech(
                base_url=auth.url, api_key=auth.apikey.get_secret_value(), model=auth.model,
                voice=auth.voice, text=cleaned, timeout=VOICE_SYNTHESIS_TIMEOUT,
            ), timeout=VOICE_SYNTHESIS_TIMEOUT)
            if generation != self.runtime.data_generation:
                return None
            return store_voice(self.runtime.config.data_dir, audio_bytes, cleaned)
        except Exception as exc:
            # Do not log raw provider errors, keys or the user's transcript.
            logger.warning("Voice synthesis fell back to text (%s)", type(exc).__name__)
            return None
