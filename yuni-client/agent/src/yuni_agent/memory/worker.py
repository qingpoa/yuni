from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import or_, select

from yuni_agent.memory.models import MemoryCandidate
from yuni_agent.memory.writer import LOCAL_TIMEZONE, save_candidate, stored_utc
from yuni_agent.model.client import ModelClient, OpenAICompatibleClient
from yuni_agent.storage.models import (
    ChatMessage,
    Conversation,
    Memory,
    MemoryEntity,
    MemoryJob,
    Settings,
)
from yuni_agent.storage.models import (
    MemoryCandidate as StoredMemoryCandidate,
)

if TYPE_CHECKING:
    from yuni_agent.runtime import AgentRuntime

logger = logging.getLogger(__name__)


class MemoryWorker(Protocol):
    async def run_once(self) -> bool:
        """Process one eligible job and return whether work was claimed."""
        ...


class MemoryMaintenanceWorker:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run_forever(self) -> None:
        while True:
            try:
                worked = await self.run_once()
                await asyncio.sleep(0.25 if worked else 5)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Memory maintenance loop failed")
                await asyncio.sleep(10)

    async def run_once(self, model_client: ModelClient | None = None) -> bool:
        if not self.runtime.memory_generation_enabled:
            return False
        client = model_client or self._configured_client()
        if client is None:
            return False
        async with self.runtime.maintenance_lock:
            job_id = self._claim_job()
            if job_id is None:
                return False
            return await self._run_claimed_job(job_id, client)

    def configured_client(self) -> ModelClient | None:
        if self.runtime.model_api_key is None:
            return None
        with self.runtime.require_database().session() as session:
            settings = session.get(Settings, 1)
            assert settings is not None
            return OpenAICompatibleClient(
                base_url=settings.model_base_url,
                api_key=self.runtime.model_api_key.get_secret_value(),
                model=settings.chat_model,
            )

    _configured_client = configured_client

    async def process_through_locked(
        self, conversation_id: int, to_message_id: int, client: ModelClient
    ) -> bool:
        """Extract pending long-term memories through a boundary while caller holds the lock."""
        if not self.runtime.memory_generation_enabled:
            return True
        job_id = self._claim_or_create_through(conversation_id, to_message_id)
        if job_id is None:
            return True
        return await self._run_claimed_job(job_id, client)

    async def _run_claimed_job(self, job_id: int, client: ModelClient) -> bool:
        try:
            await self._process_job(job_id, client)
        except Exception as exc:
            logger.warning("Memory job %s failed: %s", job_id, exc)
            self._mark_failed(job_id, exc)
            return False
        return True

    def _claim_job(self) -> int | None:
        now = datetime.now(UTC)
        with self.runtime.require_database().session() as session:
            job = session.scalar(
                select(MemoryJob)
                .where(MemoryJob.status == "pending", MemoryJob.run_after <= now)
                .order_by(MemoryJob.run_after, MemoryJob.id)
                .limit(1)
            )
            if job is None:
                return None
            conversation = session.get(Conversation, job.conversation_id)
            if conversation is not None:
                job.from_message_id = max(
                    job.from_message_id, (conversation.memory_processed_until_message_id or 0) + 1
                )
            job.status = "running"
            job.updated_at = now
            session.flush()
            return job.id

    def _claim_or_create_through(self, conversation_id: int, to_message_id: int) -> int | None:
        now = datetime.now(UTC)
        with self.runtime.require_database().session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            processed = conversation.memory_processed_until_message_id or 0
            if processed >= to_message_id:
                return None
            job = session.scalar(
                select(MemoryJob)
                .where(
                    MemoryJob.conversation_id == conversation_id,
                    MemoryJob.status == "pending",
                )
                .order_by(MemoryJob.id)
                .limit(1)
            )
            if job is None:
                job = MemoryJob(
                    conversation_id=conversation_id,
                    from_message_id=processed + 1,
                    to_message_id=to_message_id,
                    run_after=now,
                    status="running",
                )
                session.add(job)
            else:
                job.from_message_id = min(job.from_message_id, processed + 1)
                job.to_message_id = max(job.to_message_id, to_message_id)
                job.status = "running"
                job.updated_at = now
            session.flush()
            return job.id

    async def _process_job(self, job_id: int, client: ModelClient) -> None:
        generation = self.runtime.data_generation
        with self.runtime.require_database().session() as session:
            job = session.get(MemoryJob, job_id)
            if job is None or job.status != "running":
                return
            conversation = session.get(Conversation, job.conversation_id)
            if conversation is None:
                return
            companion_id = conversation.companion_id
            # Include delivered assistant context, but only new messages may be evidence.
            previous_ids = select(ChatMessage.id).where(
                ChatMessage.conversation_id == job.conversation_id,
                ChatMessage.status == "succeeded", ChatMessage.id < job.from_message_id,
            ).order_by(ChatMessage.id.desc()).limit(6)
            messages = list(session.scalars(select(ChatMessage).where(
                ChatMessage.conversation_id == job.conversation_id,
                ChatMessage.status == "succeeded", ChatMessage.id <= job.to_message_id,
                or_(ChatMessage.id >= job.from_message_id, ChatMessage.id.in_(previous_ids)),
            ).order_by(ChatMessage.id)))
            new_ids = {row.id for row in messages if row.id >= job.from_message_id}
            extraction_input = [{"id": row.id, "role": row.role, "content": row.context_content,
                "sent_at": stored_utc(row.created_at).astimezone(LOCAL_TIMEZONE).isoformat(),
                "extract": row.id in new_ids} for row in messages]
            entities = {row.id: row.canonical_name for row in session.scalars(select(MemoryEntity).where(
                MemoryEntity.companion_id == companion_id))}
            catalog = [{"id": row.id, "subject_person": entities.get(row.subject_entity_id, "user"),
                "subject_key": row.subject_key, "qualifier": row.qualifier, "kind": row.kind,
                "domain": row.domain, "scene_key": row.scene_key, "content": row.content,
                "status": row.status, "time_meta": row.time_meta} for row in session.scalars(
                select(Memory).where(Memory.companion_id == companion_id,
                    Memory.status.in_(["active", "conflicted", "deleted"])).order_by(Memory.id.desc()).limit(150))]
        candidates = []
        if new_ids:
            result = await client.complete_json(_extraction_prompt(extraction_input, catalog))
            candidates = _parse_candidates(result, new_ids)
        self._save_candidates(job_id, companion_id, candidates, generation)
        logger.info("Memory job %s accepted %s candidates", job_id, len(candidates))

    def _save_candidates(self, job_id: int, companion_id: int,
                         candidates: list[MemoryCandidate], generation: int | None = None) -> None:
        with self.runtime.embedding_lock, self.runtime.require_database().session() as session:
            job = session.get(MemoryJob, job_id)
            if job is None or job.status != "running":
                return
            conversation = session.get(Conversation, job.conversation_id)
            if conversation is None:
                return
            if (generation is not None and generation != self.runtime.data_generation) or (
                (conversation.memory_processed_until_message_id or 0) >= job.to_message_id
            ):
                job.status = "cancelled"
                return
            boundary = conversation.memory_processed_until_message_id or 0
            for candidate in candidates:
                candidate = candidate.model_copy(update={"source_message_ids": [
                    item for item in candidate.source_message_ids if boundary < item <= job.to_message_id]})
                if not candidate.source_message_ids:
                    continue
                if candidate.sensitivity == "sensitive":
                    # Validate evidence before allowing a candidate to enter the inbox.
                    from yuni_agent.memory.writer import _sources
                    try:
                        _sources(session, companion_id, candidate, job.conversation_id, False)
                    except ValueError:
                        continue
                    session.add(StoredMemoryCandidate(
                        job_id=job_id, type=candidate.type.value, subject_key=candidate.subject_key,
                        title=candidate.title, content=candidate.content, importance=candidate.importance,
                        confidence=candidate.confidence, sensitivity="sensitive",
                        source_message_ids=candidate.source_message_ids,
                        structured_data=candidate.model_dump(mode="json"), status="pending",
                        reason="敏感记忆需要用户确认"))
                else:
                    try:
                        save_candidate(session, companion_id, candidate, conversation_id=job.conversation_id)
                    except ValueError:
                        continue
            conversation.memory_processed_until_message_id = max(boundary, job.to_message_id)
            job.status = "succeeded"
            job.error_message = None

    def _mark_failed(self, job_id: int, exc: Exception) -> None:
        with self.runtime.require_database().session() as session:
            job = session.get(MemoryJob, job_id)
            if job is None or job.status == "cancelled":
                return
            job.retry_count += 1
            job.error_message = str(exc)[:500]
            if job.retry_count >= 3:
                job.status = "failed"
            else:
                retry_at = datetime.now(UTC) + timedelta(
                    minutes=2**job.retry_count
                )
                successor = session.scalar(select(MemoryJob).where(
                    MemoryJob.conversation_id == job.conversation_id,
                    MemoryJob.status == "pending", MemoryJob.id != job.id,
                ))
                if successor is not None:
                    successor.from_message_id = min(successor.from_message_id, job.from_message_id)
                    successor.to_message_id = max(successor.to_message_id, job.to_message_id)
                    successor.retry_count = max(successor.retry_count, job.retry_count)
                    successor.error_message = job.error_message
                    successor.run_after = retry_at
                    job.status = "failed"
                else:
                    job.status = "pending"
                    job.run_after = retry_at


def _extraction_prompt(messages: list[dict[str, Any]], catalog: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "结构化记忆整理：输入消息 sent_at 是相对时间唯一锚点（Asia/Shanghai），不可用执行日期。"
                "kind=fact/event/ongoing_state；domain=real_world/conversation/fictional，虚构必须有 scene_key。"
                "subject_person=user 或实体规范名；entities=[{name,entity_type,relation_to_user,aliases}]，"
                "entity_type 仅限 person/pet/organization/place/other。"
                "复用已有目录 subject_key、qualifier；同名不同关系的人不能合并。"
                "qualifier 仅在 operation_hint=coexist（同一主题在不同客观场景或时间条件下并存，如“工作日/周末”“火锅/凉拌”）时填写简短条件词，其余所有情况（含 create/transition/correct）qualifier 必须为空字符串 \"\"，严禁把喜恶极性或新旧状态（如 like/dislike/current_like/past）填入 qualifier。"
                "operation_hint=create/reinforce/enrich/transition/correct/coexist/dispute；"
                "只有同一次事件进展/状态更新（如计划取消、完成）才用 enrich 并输出 target_memory_id；不同日期的新事件 create。"
                "同结论复述输出 target_memory_id 和 same_conclusion=true；须核对完整对象及结论，"
                "不可用相似度或是否含不字判断；墓碑同样核对，无法确认则 false。"
                "事实、偏好或近况发生前后变化（如以前不喜欢吃某食物现在喜欢吃了、搬家、换工作）用 transition 并输出对应的旧记录 target_memory_id；新记忆的 content 只陈述当前生效的新事实（如“用户现在挺喜欢吃香菜。”），旧事实会由系统自动归档为历史，不要在新 content 里重复拼接旧结论。明确纠正口误用 correct，条件不同用 coexist，无法判定矛盾用 dispute。"
                "输出 evidence_quote 逐字引文；只有 extract=true 消息能作为 source_message_ids。"
                "助手只作消歧上文；已交付的承诺可作为 conversation 范围 event，不能当现实事实。"
                "时间字段：time_text,time_precision(仅限 exact/day/window/recurring/unknown),valid_from,valid_until,time_meta；"
                "time_meta={start_window_end,end_window_start,planned_start,planned_end,actual_start,actual_end}。"
                "有效期只记事实生效/明确结束；模糊窗口不能当结束。event_status 仅在 kind=event 时填写"
                "(仅限 planned/in_progress/completed/cancelled，非 event 填 null)、outcome，"
                "计划到期不代表完成。近况 fresh_days=5，持续的工作情绪压力14；不能推断康复。"
                "你负责从用户原话提取长期记忆。判断金标准：仅提取【一周或一个月后聊起新话题时，伴侣仍需要记得】的跨周期高价值信息（稳定个人背景/事实、长期偏好/忌口、重要人生或近期事件、近几天持续的身心状态、明确相处边界），每条只表达一个事实，importance 必须 >= 3（1～2分的琐碎小事直接丢弃不输出）。"
                "【严格不提取的5类噪音（直接跳过，绝不输出）】：1) 情侣日常甜言蜜语、随口夸奖与哄人场面话（如“想你了”“想到你就开心”“你越来越可爱”“没你在身边不快活”“不嫌你话多”等）；2) 一次性的文件读写、修改文本、发/换表情包等工具操作或聊天内小任务；3) 单次犯懒、某顿饭吃什么、当下困了饿了等即时短暂状态（除非明确是长期稳定的饮食习惯或忌口）；4) 顺着对方话头的玩笑接梗、自嘲或斗嘴反话；5) 与 existing_memories 结论实质重复的同义衍生（如已有自称偏好，不再拆出“觉得该自称很可爱”）。"
                "用户消息中的[发送了表情包...]仅代表聊天情绪象征或顺手活跃气氛，其画面动作、物品、场景及图中文字绝不能作为用户的现实事实、事件或近况提取。"
                "用户明确希望保存的事实或偏好应认真提取，不把‘记住’命令本身当成记忆。"
                "用户明确要求不要保存的内容不提取。询问保存状态不构成新事实。"
                "type 只能是 user_fact、user_preference、important_event、relationship_preference、"
                "emotion_pattern 或 boundary。现实/虚构事实只引用用户消息 id；"
                "conversation 事件仅限跨日仍需跟进的重要约定，绝不记录单次文件操作或斗嘴互动。"
                "返回 JSON 对象：{\"memories\":[{\"type\":\"user_preference\","
                "\"kind\":\"fact\",\"domain\":\"real_world\",\"scene_key\":null,\"subject_person\":\"user\","
                "\"subject_key\":\"food.cilantro\",\"qualifier\":\"\",\"title\":\"...\",\"content\":\"...\","
                "\"importance\":3到5,\"confidence\":0到1,\"sensitivity\":\"normal或sensitive\","
                "\"source_message_ids\":[数字],\"evidence_quote\":\"逐字引文\",\"entities\":[],"
                "\"operation_hint\":\"create\",\"target_memory_id\":null,\"same_conclusion\":false,"
                "\"time_text\":null,\"time_precision\":\"unknown\",\"valid_from\":null,\"valid_until\":null,"
                "\"time_meta\":{},\"event_status\":null,\"outcome\":null,\"fresh_days\":5}]}。"
                "按候选补充实际时间、事件和实体字段；日常闲聊无高价值长期记忆时，必须返回 memories 空数组。"
            ),
        },
        {"role": "user", "content": json_dumps({"messages": messages, "existing_memories": catalog or []})},
    ]


def _parse_candidates(result: dict[str, Any], allowed_ids: set[int]) -> list[MemoryCandidate]:
    if "items" in result and "memories" not in result:
        result = {**result, "memories": result["items"]}
    if "memories" not in result:
        raise ValueError("memory response is missing memories")
    raw_items = result["memories"]
    if not isinstance(raw_items, list):
        raise ValueError("memories must be a list")
    candidates: list[MemoryCandidate] = []
    invalid_schema = 0
    for raw in raw_items[:10]:
        if not isinstance(raw, dict):
            invalid_schema += 1
            continue
        try:
            candidate = MemoryCandidate.model_validate(raw)
        except ValueError:
            invalid_schema += 1
            continue
        source_ids = list(dict.fromkeys(candidate.source_message_ids))
        if (
            not source_ids
            or not set(source_ids) <= allowed_ids
            or candidate.confidence < 0.8
            or candidate.importance < 3
        ):
            continue
        if candidate.sensitivity not in {"normal", "sensitive"}:
            continue
        candidates.append(candidate.model_copy(update={"source_message_ids": source_ids}))
    if raw_items and invalid_schema == len(raw_items[:10]):
        raise ValueError("memory candidates failed schema validation")
    return candidates


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
