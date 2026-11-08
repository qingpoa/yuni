from __future__ import annotations

import logging
import math
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select

from yuni_agent.context.budget import estimate_tokens
from yuni_agent.memory.index import MemoryIndex, active_memories, content_hash
from yuni_agent.memory.models import RetrievedMemory
from yuni_agent.memory.writer import LOCAL_TIMEZONE, stored_utc
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import Memory, MemoryEntity

logger = logging.getLogger(__name__)


def parse_time_bound(value: str | None, *, end: bool = False) -> datetime | None:
    if value is None:
        return None
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        result = datetime.fromisoformat(value).replace(tzinfo=LOCAL_TIMEZONE)
        return result + timedelta(days=1) if end else result
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('时间须为 ISO 日期或带时区的时间')
    return result


def _timestamp(value: datetime) -> float:
    return stored_utc(value).timestamp()


def _tokens(value: str) -> list[str]:
    values = []
    for part in re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]+', value.lower()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', part):
            values.extend(part[i:i+2] for i in range(max(1, len(part)-1)))
        else:
            values.append(part)
    return values


def _terms(value: str) -> set[str]:
    return set(_tokens(value))


def _bm25(query: str, texts: dict[int, str]) -> dict[int, float]:
    documents = {key: Counter(_tokens(text)) for key, text in texts.items()}
    terms = _terms(query)
    average = sum(sum(doc.values()) for doc in documents.values()) / max(1, len(documents)) or 1
    frequency = {term: sum(term in doc for doc in documents.values()) for term in terms}
    scores = {}
    for key, doc in documents.items():
        length = sum(doc.values())
        score = 0.0
        for term in terms:
            tf = doc.get(term, 0)
            if tf:
                idf = math.log(1 + (len(documents) - frequency[term] + .5) / (frequency[term] + .5))
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / average))
        if score:
            scores[key] = score
    return scores


def _time_overlap(row: Memory, start: datetime | None, end: datetime | None) -> tuple[bool, bool]:
    meta = row.time_meta or {}
    if row.kind == 'event':
        first = meta.get('actual_start') or meta.get('planned_start')
        last = meta.get('actual_end') or meta.get('planned_end')
        left = datetime.fromisoformat(first) if first else None
        right = datetime.fromisoformat(last) if last else None
        # A known start without an end is a point, not an infinite event.
        if left and right is None:
            right = left + timedelta(microseconds=1)
        if left is None and right is None:
            return start is None and end is None, False
    else:
        left, right = row.valid_from, row.valid_until
    overlaps = (end is None or left is None or _timestamp(left) < _timestamp(end)) and (
        start is None or right is None or _timestamp(start) < _timestamp(right))
    uncertain = False
    for first, last in ((row.valid_from, meta.get('start_window_end')), (meta.get('end_window_start'), row.valid_until)):
        if first and last:
            a = datetime.fromisoformat(first) if isinstance(first, str) else first
            b = datetime.fromisoformat(last) if isinstance(last, str) else last
            if (end is None or _timestamp(a) < _timestamp(end)) and (start is None or _timestamp(start) < _timestamp(b)):
                uncertain = True
    return overlaps, uncertain


class MemoryRetriever(Protocol):
    async def retrieve(self, companion_id: int, query: str, limit: int = 6,
                       token_budget: int = 1500, **filters) -> list[RetrievedMemory]: ...


class SqliteMemoryRetriever:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def retrieve(self, companion_id: int, query: str, limit: int = 6, token_budget: int = 1500, *,
                       memory_type: str = '', updated_after: datetime | None = None,
                       updated_before: datetime | None = None, kind: str = 'all', include_history: bool = False,
                       time_from: str | None = None, time_to: str | None = None, entity_name: str = '',
                       domain: str = 'real_world', scene_key: str | None = None) -> list[RetrievedMemory]:
        if not self.runtime.memory_retrieval_enabled:
            return []
        start, end = parse_time_bound(time_from), parse_time_bound(time_to, end=True)
        if start and end and start >= end:
            raise ValueError('时间范围无效')
        generation, embedding_generation = self.runtime.data_generation, self.runtime.embedding_generation
        with self.runtime.require_database().session() as session:
            snapshot_hashes = {item.id: content_hash(item) for item in active_memories(session, companion_id, include_history=True)}
        try:
            semantic = await MemoryIndex(self.runtime).search(companion_id, query, limit=100) if query.strip() else {}
        except Exception:
            logger.warning('长期记忆语义检索不可用，使用文本检索')
            semantic = {}
        if generation != self.runtime.data_generation or not self.runtime.memory_retrieval_enabled:
            return []
        now = datetime.now(UTC)
        historical = include_history or start is not None or end is not None
        with self.runtime.require_database().session() as session:
            all_rows = active_memories(session, companion_id, include_history=True)
            rows = list(all_rows)
            entities = list(session.scalars(select(MemoryEntity).where(MemoryEntity.companion_id == companion_id, MemoryEntity.status == 'active')))
        predecessors: dict[int, list[Memory]] = {}
        for item in all_rows:
            if item.status == 'historical' and item.superseded_by_id is not None:
                predecessors.setdefault(item.superseded_by_id, []).append(item)
        names = {item.id: [item.canonical_name, *item.aliases] for item in entities}
        entity_filter = {key for key, aliases in names.items() if any(entity_name.lower() in name.lower() for name in aliases)} if entity_name else set()
        filtered = []
        uncertainty = {}
        for row in rows:
            if row.domain != domain or (domain == 'fictional' and row.scene_key != scene_key):
                continue
            if not historical and row.status != 'active':
                continue
            if kind != 'all' and row.kind != kind or memory_type and row.type != memory_type:
                continue
            if entity_name and not entity_filter.intersection([*row.entity_ids, row.subject_entity_id]):
                continue
            if updated_after and _timestamp(row.updated_at) < _timestamp(updated_after) or updated_before and _timestamp(row.updated_at) > _timestamp(updated_before):
                continue
            if not historical:
                if row.kind != 'event' and not _time_overlap(row, now, now + timedelta(microseconds=1))[0]:
                    continue
                if row.kind == 'ongoing_state' and (row.fresh_until is None or _timestamp(row.fresh_until) <= now.timestamp()):
                    continue
                if row.kind == 'event':
                    date = row.time_meta.get('actual_start') or row.time_meta.get('planned_start')
                    event_time = _timestamp(datetime.fromisoformat(date)) if date else _timestamp(row.created_at)
                    if event_time < (now-timedelta(days=30)).timestamp():
                        continue
            matches, uncertain = _time_overlap(row, start, end)
            if not matches:
                continue
            filtered.append(row)
            uncertainty[row.id] = uncertain
        rows = filtered
        semantic = {key: score for key, score in semantic.items() if any(
            row.id == key and snapshot_hashes.get(key) == content_hash(row) for row in rows)
            and embedding_generation == self.runtime.embedding_generation}
        texts = {row.id: f'{row.title} {row.content} {row.subject_key} {row.qualifier} {row.outcome or ""} {row.time_text or ""}' for row in rows}
        lexical = _bm25(query, texts)
        overview = not query.strip() or bool(re.search(
            r'(?:记住|记下(?:来)?|记得)(?:了|的|我|关于我)?(?:什么|哪些|啥|哪几)|(?:有哪些|列出|看看)(?:关于我|我的)?(?:长期)?记忆', query))
        if overview:
            lexical = {row.id: lexical.get(row.id, 0) for row in rows}
        alias_scores = {}
        for row in rows:
            aliases = [name for key in [*row.entity_ids, row.subject_entity_id] for name in names.get(key, [])]
            count = sum(name.lower() in query.lower() for name in aliases if name)
            if count:
                alias_scores[row.id] = count
        ranks = [{key: rank for rank, (key, _) in enumerate(sorted(scores.items(), key=lambda pair: pair[1], reverse=True), 1)} for scores in (semantic, lexical, alias_scores)]
        ranked = []
        for row in rows:
            score = sum(1/(60+rank[row.id]) for rank in ranks if row.id in rank)
            fresh_bonus = int(row.kind == 'ongoing_state' and row.fresh_until is not None and _timestamp(row.fresh_until) > now.timestamp())
            if score or (fresh_bonus and not historical):
                planned_bonus = int(row.kind == 'event' and row.event_status == 'planned')
                ranked.append((score, fresh_bonus+planned_bonus, row))
        ranked.sort(key=lambda value: (value[0], value[1], value[2].importance, value[2].confidence, _timestamp(value[2].updated_at), value[2].id), reverse=True)
        results, used_tokens = [], 0
        for _, _, row in ranked:
            prev_facts: list[str] = []
            if row.status == 'active':
                visited: set[int] = {row.id}
                queue = list(predecessors.get(row.id, []))
                while queue and len(prev_facts) < 3:
                    old = queue.pop(0)
                    if old.id in visited:
                        continue
                    visited.add(old.id)
                    if old.content and old.content not in prev_facts:
                        prev_facts.append(old.content)
                    queue.extend(predecessors.get(old.id, []))
            result = RetrievedMemory(id=row.id, type=row.type, content=row.content, importance=row.importance,
                confidence=row.confidence, kind=row.kind, domain=row.domain, status=row.status,
                subject_key=row.subject_key, qualifier=row.qualifier, event_status=row.event_status,
                outcome=row.outcome, time_text=row.time_text, valid_from=row.valid_from, valid_until=row.valid_until,
                fresh_until=row.fresh_until, time_meta=row.time_meta,
                time_certainty='uncertain_window' if uncertainty.get(row.id) else None,
                previous_facts=prev_facts)
            cost = estimate_tokens(result.model_dump_json()) + 8
            if used_tokens+cost > token_budget:
                continue
            results.append(result)
            used_tokens += cost
            if len(results) >= limit:
                break
        return results
