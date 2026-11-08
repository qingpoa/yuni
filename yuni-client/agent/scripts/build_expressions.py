"""Offline high-purity micro-reaction expression library builder.

Builds a ~280-item human conversational micro-reaction library (2~9 chars, zero ``xxxx``
templates, zero plot-hijacking sentences, zero AI-fabricated lines) by:
  1. Cleaning & preserving genuine user-chat sourced expressions in ``yuni.db``;
  2. Salvaging verbatim human micro-reactions from ``phrase_cleaned_800.json`` via strict
     hard rules + LLM private-chat naturalness audit;
  3. Mining additional verbatim 2~9 char micro-reactions from private-style short dialogues
     in ``lccc_base_valid.jsonl.gz``, double-auditing, and deduplicating by normalized text
     and embedding cosine similarity (< 0.85).

Usage (from project root or yuni-client/):
  python yuni-client/agent/scripts/build_expressions.py --target 280
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sqlalchemy import delete, func, select

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[0] / "src"))

from yuni_agent.expressions.index import content_hash
from yuni_agent.model.auth import load_auth
from yuni_agent.model.embedding import EmbeddingClient
from yuni_agent.storage.database import Database
from yuni_agent.storage.models import (
    Companion,
    Expression,
    ExpressionSource,
    ExpressionVector,
)

PUBLIC_AND_DATED_NOISE = (
    "转发微博", "博主", "楼主", "私信", "粉丝", "超话", "爱豆", "打卡", "互关", "抽奖",
    "代购", "包邮", "链接", "求粉", "赞我", "点赞", "评论区", "小编", "原博", "up主",
    "群主", "进群", "加微信", "朋友圈", "驾校", "住院", "赔偿", "相亲", "婚礼", "婆婆",
    "儿媳", "集福", "敬业福", "http", "www",
    "肿么", "神马", "酱紫", "偶们", "稀饭", "蓝瘦", "香菇", "表这样", "泥垢", "桑心",
    "伦家", "银家", "矮油", "灰常", "木有", "森么", "捉急", "喜大普奔", "不明觉厉", "么么哒",
    "亲，", "吧亲", "==",
)

PHRASE_BAN_SUBSTRINGS = (
    "xxxx", "==", "亲", "您好", "老板", "博主", "楼主", "粉丝", "私聊", "群", "闺蜜",
    "姐姐", "哥哥", "妹妹", "弟弟", "师姐", "师兄", "阿姨", "叔叔", "妈妈", "爸爸",
    "老汉", "闺女", "儿子", "室友", "舍友", "寝室", "同学", "老师", "教练", "朋友",
    "高考", "中考", "开学", "放假", "考完", "作业", "工资", "红包", "封包", "票子",
    "屁", "要脸", "脸是什么", "活着", "砸墙", "砸门", "砸了", "死", "滚", "病", "抑郁",
    "肥来", "摸港", "晓得", "好久", "介样", "肿么", "怎么肥四", "肥四", "木有", "心水",
    "债见", "问哈", "嗯哪", "嗯呐", "么么哒", "嘻嘻", "呵呵", "摸头", "苦力", "打工",
    "请假", "组织", "罐头", "瑜伽", "衣服", "端杯", "高雅", "看望", "露个头", "狗抢",
    "被偷", "太贵", "双字", "歌词", "阳台", "火候", "做菜", "真爱", "心如止水", "青春无价",
    "人生已足以", "命运的安排", "快活哦", "有车", "种花", "音频", "网名", "紧身", "水分",
    "端走", "端空气", "不买", "鞋好看", "贼沉", "九点可不行", "顶多三天", "是愤慨吧",
    "对话说什么", "老了", "潜水太久", "鸟枪换炮", "蜜汁优越", "包的真漂亮", "在赖一晚",
)

INTIMATE_DIALOGUE_CUES = (
    "想你", "抱抱", "早点睡", "晚安", "早安", "困了", "饿了", "吃什么", "好吃", "不想动",
    "好累", "哄", "生气", "不理你", "笨", "讨厌", "喜欢", "开心", "委屈", "撒娇",
    "陪我", "在干嘛", "忙完", "下班", "起床", "睡醒", "做梦", "熬夜", "心疼", "乖",
    "骗你", "真的吗", "算了", "随便", "听你的", "服了",
)

EXTRACT_MICRO_PROMPT = (
    "你负责从多组真实人类日常短对话中，提取适合【一对一微信伴侣聊天】复用的原话短促口语反应。\n"
    "输入包含若干组独立对话（每组含 dlg_id 与 turns 列表）。请只挑选完全满足以下硬标准的短句：\n"
    "1. 逐字原话：style 必须是某条 turn 原文中的完整短句或独立分句（2～9个汉字，含必要末尾标点不超过10字符），保留原字原语气，严禁改写、严禁编造、严禁使用 xxxx 挖空！\n"
    "2. 纯反应态、零剧情绑架：只提取表达情绪、态度、语气接球、轻度调侃、撒娇、嘴硬、妥协、安慰、反问、好奇追问、犯困嘴馋或打招呼的纯口语短反应（例如：“那就行”“怪我？”“故意的”“没事慢慢来”“干嘛这样说”“想得美”“那倒也是”“困死了”）。\n"
    "3. 严格跳过：凡含具体人物称谓、地名、特定物品、具体事件陈述、数字时间、古早网络词（呵呵/嘻嘻/么么哒/亲/肿么）、粗俗攻击词或生僻方言的句子，一律跳过！\n"
    "4. situation 写明【伴侣在什么互动情境下用这句话回应对方】（4～16字，立场必须是说话人回应对方）。\n"
    "宁缺毋滥，每组对话提取 0～3 条即可。严格只输出 JSON 对象：\n"
    '{"items":[{"dlg_id":0,"turn_idx":0,"situation":"...","style":"..."}]}'
)

STRICT_AUDIT_PROMPT = (
    "你是2026年现代年轻伴侣微信一对一私聊口语库的严格终审员。输入为候选列表，每项含 idx、situation、style、source_text。\n"
    "请逐条判断该短语放进通用恋爱伴侣口语库是否【自然、鲜活、高泛用且零剧情污染】：\n"
    "【通过（keep）必须同时满足】：\n"
    "1. 纯粹的情绪/语气/互动短反应（2～8字最佳）：如随口应和、撒娇、嘴硬、轻度打趣、反问、安慰、犯困、嘴馋、想念、妥协、惊喜、打招呼。\n"
    "2. 现代微信活人感：拒绝古早网络词（如“木有/肥四/心水/嘻嘻/呵呵/么么哒/鸟枪换炮/蜜汁”）、拒绝客服腔、拒绝老气相声捧哏腔、拒绝错别字或不通顺残句。\n"
    "3. 绝对零具体事物与零线下特定剧情：凡提到具体物品（鞋、花、车、音频、装备、饺子等）、具体时间约定（后天、一周、一年、三天、九点、周末等）、盘问对方隐私/行程（你在哪儿、去哪潇洒、找工作、减肥等）或自述特定线下动作（我到家了、我走了、帮你拍等），一律 reject！\n"
    "4. 角色通用可塑：不粗俗不伤人，不带固定性别自称；允许在 keep 时将 refined_situation 微调为精准的【伴侣回应情境】（4～14字），严禁修改 style！\n"
    "宁可淘汰70%也绝不放过任何一条带具体剧情或老气的句子；不满足任一条则 action 填 \"reject\"。\n"
    '严格只输出 JSON 对象：{"decisions":[{"idx":0,"action":"keep"|"reject","refined_situation":"..."}]}'
)


def strip_punct(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).casefold()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def passes_micro_rules(situation: str, style: str, source_text: str | None = None) -> bool:
    sit = situation.strip()
    sty = style.strip()
    if not (3 <= len(sit) <= 18 and 2 <= len(sty) <= 10):
        return False
    core = strip_punct(sty)
    if not (2 <= len(core) <= 9):
        return False
    if re.search(r"[0-9a-zA-Z]", sty):
        return False
    if re.search(r"[!！?？~～。.\.]{3,}", sty):
        return False
    combined = sit + sty
    if any(bad in combined for bad in PHRASE_BAN_SUBSTRINGS):
        return False
    if source_text is not None and compact(sty) not in compact(source_text):
        return False
    return True


def iter_private_lccc_dialogues(gz_path: Path):
    seen: set[str] = set()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            raw_turns = row if isinstance(row, list) else row["dialog"]
            turns = [re.sub(r"\s+", "", t) for t in raw_turns]
            if not (3 <= len(turns) <= 8):
                continue
            if any(not (2 <= len(t) <= 34) for t in turns):
                continue
            joined = " ".join(turns)
            if any(noise in joined for noise in PUBLIC_AND_DATED_NOISE):
                continue
            if not any(cue in joined for cue in INTIMATE_DIALOGUE_CUES):
                continue
            digest = hashlib.sha256(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            yield idx, turns


async def build_library(
    *,
    data_dir: Path,
    output_dir: Path,
    target_total: int = 280,
    similarity_threshold: float = 0.85,
) -> None:
    t0 = time.time()
    db_path = data_dir / "yuni.db"
    auth = load_auth(data_dir / "auth.json")
    emb_auth = load_auth(data_dir / "embedding-auth.json")
    if auth is None or emb_auth is None:
        raise SystemExit("缺少 auth.json 或 embedding-auth.json")
    emb = EmbeddingClient(emb_auth)

    # 1. Backup database before touching anything
    backup_dir = output_dir / "data-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"yuni-before-expr-280-{ts}.db"
    src_con = sqlite3.connect(db_path)
    dst_con = sqlite3.connect(backup_path)
    with dst_con:
        src_con.backup(dst_con)
    dst_con.close()
    src_con.close()
    print(f"[1/5] 已备份数据库至 {backup_path.name}", flush=True)

    # 2. Clean & load existing sourced expressions from yuni.db
    db = Database.open(db_path)
    kept_sourced: list[dict[str, Any]] = []
    try:
        with db.session() as session:
            comp = session.scalar(select(Companion).limit(1))
            if comp is None:
                raise SystemExit("数据库中尚无伴侣记录")
            comp_id = comp.id
            sourced_ids = set(session.scalars(select(ExpressionSource.expression_id).distinct()))
            all_rows = list(
                session.scalars(
                    select(Expression).where(Expression.companion_id == comp_id).order_by(Expression.id)
                )
            )
            # Remove any unsourced rows or sourced rows that contain xxxx / user-command noise
            remove_ids = [
                r.id for r in all_rows
                if r.id not in sourced_ids
                or "xxxx" in r.style
                or r.style.strip() in {"再来一张", "来个蛋，我一会要打游戏去了"}
            ]
            if remove_ids:
                session.execute(delete(ExpressionSource).where(ExpressionSource.expression_id.in_(remove_ids)))
                session.execute(delete(ExpressionVector).where(ExpressionVector.expression_id.in_(remove_ids)))
                session.execute(delete(Expression).where(Expression.id.in_(remove_ids)))
                session.flush()

            clean_rows = list(
                session.scalars(
                    select(Expression).where(Expression.companion_id == comp_id).order_by(Expression.id)
                )
            )
            for row in clean_rows:
                kept_sourced.append({"id": row.id, "situation": row.situation, "style": row.style})
    finally:
        db.close()

    # Embed kept sourced items to seed the deduplication matrix
    sourced_texts = [f"情境：{it['situation']}\n表达方式：{it['style']}" for it in kept_sourced]
    sourced_vecs: list[list[float]] = []
    for i in range(0, len(sourced_texts), 16):
        sourced_vecs.extend(await emb.embed(sourced_texts[i : i + 16]))

    seen_cores: set[str] = {strip_punct(it["style"]) for it in kept_sourced}
    matrix = np.asarray(sourced_vecs, dtype="float32") if sourced_vecs else None
    curated: list[dict[str, Any]] = []
    needed = max(0, target_total - len(kept_sourced))
    print(f"[2/5] 保留真实聊天纯净表达 {len(kept_sourced)} 条，目标新增精锐真人短反应 {needed} 条（总目标 {target_total} 条）", flush=True)

    sem = asyncio.Semaphore(8)
    llm_calls = 0
    deepseek = auth.model.lower().split("/")[-1].startswith("deepseek")

    async with httpx.AsyncClient(timeout=60.0) as http:
        async def ask_json(messages: list[dict[str, str]], temp: float = 0.15) -> dict[str, Any]:
            nonlocal llm_calls
            payload: dict[str, Any] = {
                "model": auth.model,
                "messages": messages,
                "temperature": temp,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            if deepseek:
                payload["thinking"] = {"type": "disabled"}
            async with sem:
                for attempt in range(3):
                    try:
                        llm_calls += 1
                        resp = await http.post(
                            auth.url.rstrip("/") + "/chat/completions",
                            headers={"Authorization": f"Bearer {auth.apikey.get_secret_value()}"},
                            json=payload,
                        )
                        resp.raise_for_status()
                        content = resp.json()["choices"][0]["message"]["content"].strip()
                        if content.startswith("```"):
                            content = content.strip("`").removeprefix("json").strip()
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        if attempt == 2:
                            return {}
                        await asyncio.sleep(1.0)
            return {}

        async def audit_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not candidates:
                return []
            chunks = [candidates[i : i + 25] for i in range(0, len(candidates), 25)]

            async def run_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
                view = [
                    {
                        "idx": i,
                        "situation": c["situation"],
                        "style": c["style"],
                        "source_text": c["source_text"],
                    }
                    for i, c in enumerate(chunk)
                ]
                res = await ask_json(
                    [
                        {"role": "system", "content": STRICT_AUDIT_PROMPT},
                        {"role": "user", "content": json.dumps(view, ensure_ascii=False)},
                    ],
                    temp=0.1,
                )
                decisions = {
                    d.get("idx"): d for d in res.get("decisions", []) if isinstance(d, dict)
                }
                approved: list[dict[str, Any]] = []
                for i, c in enumerate(chunk):
                    d = decisions.get(i)
                    if not d or d.get("action") != "keep":
                        continue
                    new_sit = str(d.get("refined_situation") or c["situation"]).strip()
                    if not passes_micro_rules(new_sit, c["style"], c["source_text"]):
                        continue
                    approved.append({**c, "situation": new_sit})
                return approved

            results = await asyncio.gather(*(run_chunk(ch) for ch in chunks))
            return [item for group in results for item in group]

        async def accept_audited(approved: list[dict[str, Any]]) -> int:
            nonlocal matrix
            fresh: list[dict[str, Any]] = []
            for item in approved:
                core = strip_punct(item["style"])
                if not core or core in seen_cores:
                    continue
                seen_cores.add(core)
                fresh.append(item)
            if not fresh:
                return 0
            texts = [f"情境：{it['situation']}\n表达方式：{it['style']}" for it in fresh]
            vecs: list[list[float]] = []
            for i in range(0, len(texts), 16):
                vecs.extend(await emb.embed(texts[i : i + 16]))

            added = 0
            for item, vec in zip(fresh, vecs, strict=True):
                if len(curated) >= needed:
                    break
                arr = np.asarray(vec, dtype="float32")
                if matrix is not None and len(matrix):
                    max_sim = float(np.max(matrix @ arr))
                    if max_sim >= similarity_threshold:
                        continue
                item["vector"] = vec
                curated.append(item)
                matrix = arr.reshape(1, -1) if matrix is None else np.vstack([matrix, arr])
                added += 1
            return added

        # 3. Stage 1: Salvage from phrase_cleaned_800.json
        stage800_path = output_dir / "phrase_cleaned_800.json"
        if stage800_path.is_file():
            raw800 = json.loads(stage800_path.read_text(encoding="utf-8")).get("items", [])
            pre_filtered = [
                {
                    "situation": it["situation"].strip(),
                    "style": it["style"].strip(),
                    "source_text": it.get("source_text", it["style"]),
                    "origin": it.get("source", "stage800"),
                }
                for it in raw800
                if passes_micro_rules(it["situation"], it["style"], it.get("source_text", it["style"]))
            ]
            approved_s1 = await audit_candidates(pre_filtered)
            added_s1 = await accept_audited(approved_s1)
            print(
                f"[3/5] 阶段一（既有缓存提纯）：初筛 {len(pre_filtered)} 条 -> 终审通过 {len(approved_s1)} 条 -> 向量去重后入池 {added_s1} 条（当前 {len(curated)}/{needed}）",
                flush=True,
            )

        # 4. Stage 2: Mine from 960 private-style dialogues in lccc_base_valid.jsonl.gz
        gz_path = output_dir / "lccc_base_valid.jsonl.gz"
        if len(curated) < needed and gz_path.is_file():
            dialogues = list(iter_private_lccc_dialogues(gz_path))
            print(f"[4/5] 阶段二（私聊语料定向提纯）：共 {len(dialogues)} 组候选真人私聊对话，继续补齐剩余 {needed - len(curated)} 条...", flush=True)

            async def extract_group(group: list[tuple[int, list[str]]]) -> list[dict[str, Any]]:
                payload = [{"dlg_id": i, "turns": turns} for i, (_, turns) in enumerate(group)]
                res = await ask_json(
                    [
                        {"role": "system", "content": EXTRACT_MICRO_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temp=0.2,
                )
                extracted: list[dict[str, Any]] = []
                for raw in res.get("items", []):
                    if not isinstance(raw, dict):
                        continue
                    did = raw.get("dlg_id")
                    tidx = raw.get("turn_idx")
                    if type(did) is not int or not (0 <= did < len(group)):
                        continue
                    orig_idx, turns = group[did]
                    sit = str(raw.get("situation") or "").strip()
                    sty = str(raw.get("style") or "").strip()
                    # Verify verbatim grounding against the specified turn or any turn in that dialogue
                    src_turn = turns[tidx] if type(tidx) is int and 0 <= tidx < len(turns) else ""
                    if compact(sty) in compact(src_turn):
                        grounded_turn = src_turn
                    else:
                        grounded_turn = next((t for t in turns if compact(sty) in compact(t)), "")
                    if not grounded_turn:
                        continue
                    if not passes_micro_rules(sit, sty, grounded_turn):
                        continue
                    extracted.append(
                        {
                            "situation": sit,
                            "style": sty,
                            "source_text": grounded_turn,
                            "origin": f"lccc-private:{orig_idx}",
                        }
                    )
                return extracted

            # Process in waves of 48 dialogues (8 concurrent calls × 6 dialogues each)
            wave_size = 48
            for cursor in range(0, len(dialogues), wave_size):
                if len(curated) >= needed:
                    break
                wave = dialogues[cursor : cursor + wave_size]
                sub_groups = [wave[i : i + 6] for i in range(0, len(wave), 6)]
                extracted_lists = await asyncio.gather(*(extract_group(g) for g in sub_groups))
                raw_candidates = [item for sub in extracted_lists for item in sub]
                # Deduplicate within wave before sending to strict audit
                unique_wave: list[dict[str, Any]] = []
                wave_cores: set[str] = set()
                for c in raw_candidates:
                    core = strip_punct(c["style"])
                    if core in seen_cores or core in wave_cores:
                        continue
                    wave_cores.add(core)
                    unique_wave.append(c)
                approved_wave = await audit_candidates(unique_wave)
                added_wave = await accept_audited(approved_wave)
                print(
                    f"  -> 扫描进度 {min(cursor + wave_size, len(dialogues))}/{len(dialogues)}：本波初提 {len(unique_wave)} -> 终审 {len(approved_wave)} -> 新增 {added_wave}（累计 {len(curated)}/{needed}，LLM调用={llm_calls}）",
                    flush=True,
                )

    # Save curated manifest snapshot for inspection / future reuse
    snapshot_path = output_dir / "curated_micro_expressions.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "embedding_model": emb.auth.model,
                "embedding_profile": emb.profile,
                "sourced_kept": len(kept_sourced),
                "curated_added": len(curated),
                "items": [
                    {
                        "situation": it["situation"],
                        "style": it["style"],
                        "source_text": it["source_text"],
                        "origin": it["origin"],
                    }
                    for it in curated
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 5. Write curated items + vectors into yuni.db and ensure all sourced vectors are current
    db = Database.open(db_path)
    try:
        with db.session() as session:
            comp = session.scalar(select(Companion).limit(1))
            assert comp is not None
            comp_id = comp.id
            # Refresh vectors for kept sourced expressions
            for src_item, vec in zip(kept_sourced, sourced_vecs, strict=True):
                row = session.get(Expression, src_item["id"])
                if row is not None:
                    session.merge(
                        ExpressionVector(
                            expression_id=row.id,
                            profile=emb.profile,
                            model=emb.auth.model,
                            dimension=len(vec),
                            content_hash=content_hash(row),
                            vector=vec,
                        )
                    )
            session.flush()

            for item in curated:
                new_row = Expression(
                    companion_id=comp_id,
                    situation=item["situation"],
                    style=item["style"],
                    count=1,
                )
                session.add(new_row)
                session.flush()
                vec = item["vector"]
                session.merge(
                    ExpressionVector(
                        expression_id=new_row.id,
                        profile=emb.profile,
                        model=emb.auth.model,
                        dimension=len(vec),
                        content_hash=content_hash(new_row),
                        vector=vec,
                    )
                )
            session.flush()

            total_expr = session.scalar(
                select(func.count()).select_from(Expression).where(Expression.companion_id == comp_id)
            )
            total_vec = session.scalar(select(func.count()).select_from(ExpressionVector))
            print(
                f"\n[5/5] 构建并入库完成！当前总表达数：{total_expr}，有效向量数：{total_vec}，总耗时：{time.time() - t0:.1f}s",
                flush=True,
            )
            print(f"审核清单已保存至：{snapshot_path}", flush=True)
    finally:
        db.close()


def main() -> None:
    root = SCRIPT_DIR.parents[2]
    parser = argparse.ArgumentParser(description="构建高纯度真人短反应表达库")
    parser.add_argument("--data-dir", type=Path, default=root / "output/first-round/data")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--target", type=int, default=280)
    parser.add_argument("--sim-threshold", type=float, default=0.85)
    args = parser.parse_args()
    asyncio.run(
        build_library(
            data_dir=args.data_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            target_total=args.target,
            similarity_threshold=args.sim_threshold,
        )
    )


if __name__ == "__main__":
    main()
