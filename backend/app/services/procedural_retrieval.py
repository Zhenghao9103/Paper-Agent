"""Local cosine retrieval over the current, instance-wide procedural skill library."""

import heapq
import logging
import math
from collections.abc import Iterable

from sqlalchemy import and_, delete, false, select, update
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.memory import ProceduralSkill
from ..models.procedural_index import SkillCluster, SkillMember
from ..rag.vector_store import embed_text

logger = logging.getLogger(__name__)


def _unit(values: list[float]) -> list[float]:
    length = math.hypot(*values)
    if not length or not math.isfinite(length):
        raise ValueError("Invalid procedural embedding")
    return [value / length for value in values]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _centers(db: Session, vector: list[float], limit: int):
    centers = db.scalars(select(SkillCluster).execution_options(populate_existing=True))
    scored = (
        (_dot(vector, _unit(row.vector_sum)), row)
        for row in centers
        if any(row.vector_sum)
    )
    return heapq.nsmallest(limit, scored, key=lambda pair: (-pair[0], pair[1].id))


def _insert(db: Session, skill: ProceduralSkill) -> None:
    vector = _unit(embed_text(skill.condition, input_type="document"))
    closest = _centers(db, vector, limit=1)
    if closest and closest[0][0] >= get_settings().procedural_cluster_threshold:
        cluster = closest[0][1]
        cluster.size += 1
        cluster.vector_sum = [
            a + b for a, b in zip(cluster.vector_sum, vector, strict=True)
        ]
    else:
        cluster = SkillCluster(size=1, vector_sum=vector)
        db.add(cluster)
        db.flush()
    db.add(
        SkillMember(
            skill_id=skill.id,
            version=skill.version,
            vector=vector,
            cluster_id=cluster.id,
        )
    )
    db.flush()


def sync_skill(db: Session, skill_id: str) -> None:
    """Apply the current source version. The worker commits this with its acknowledgement."""
    # Acquire SQLite's writer lock before reading the old centroid or source version.
    db.execute(update(SkillCluster).where(false()).values(size=SkillCluster.size))
    skill = db.get(ProceduralSkill, skill_id, populate_existing=True)
    member = db.get(SkillMember, skill_id, populate_existing=True)
    if member is not None:
        if skill is not None and skill.active and skill.version == member.version:
            return
        cluster = db.get(SkillCluster, member.cluster_id, populate_existing=True)
        db.delete(member)
        db.flush()
        if cluster is not None:
            if cluster.size == 1:
                db.delete(cluster)
            else:
                cluster.size -= 1
                cluster.vector_sum = [
                    a - b
                    for a, b in zip(cluster.vector_sum, member.vector, strict=True)
                ]
            db.flush()
    if skill is not None and skill.active:
        _insert(db, skill)


def rebuild_index(db: Session) -> None:
    """Rebuild atomically in stable source-ID order; leave outer commit to the caller."""
    # Python sqlite3 legacy transaction mode doesn't start a transaction on SAVEPOINT.
    db.execute(update(SkillCluster).where(false()).values(size=SkillCluster.size))
    with db.begin_nested():
        db.execute(delete(SkillMember))
        db.execute(delete(SkillCluster))
        live = (
            select(ProceduralSkill)
            .where(ProceduralSkill.active)
            .order_by(ProceduralSkill.id)
        )
        for skill in db.scalars(live.execution_options(populate_existing=True)):
            _insert(db, skill)


def _rank(rows: Iterable, query: list[float]) -> list[ProceduralSkill]:
    threshold = get_settings().procedural_recall_threshold

    def hits():
        for skill, vector in rows:
            if vector is None:
                vector = _unit(embed_text(skill.condition, input_type="document"))
            score = _dot(vector, query)
            if score >= threshold:
                yield score, skill

    best = heapq.nsmallest(4, hits(), key=lambda pair: (-pair[0], pair[1].id))
    return [skill for _, skill in best]


def _current_members():
    return and_(
        SkillMember.skill_id == ProceduralSkill.id,
        SkillMember.version == ProceduralSkill.version,
    )


def _flat(db: Session, query: list[float], use_index: bool) -> list[ProceduralSkill]:
    live = (
        select(ProceduralSkill)
        .where(ProceduralSkill.active)
        .execution_options(populate_existing=True)
    )
    if use_index:
        rows = db.execute(
            live.add_columns(SkillMember.vector).outerjoin(
                SkillMember, _current_members()
            )
        )
    else:
        rows = ((skill, None) for skill in db.scalars(live))
    return _rank(rows, query)


def retrieve_skills(
    db: Session, question: str, mode: str
) -> tuple[list[ProceduralSkill], bool]:
    if mode not in {"flat", "rsi"} or not question.strip():
        return [], False
    live = select(ProceduralSkill.id).where(ProceduralSkill.active).limit(1)
    if db.scalar(live) is None:
        return [], False
    query = _unit(embed_text(question, input_type="query"))
    try:
        if mode == "flat":
            return _flat(db, query, use_index=True), False
        if db.scalar(select(SkillMember.skill_id).limit(1)) is None:
            rebuild_index(db)
        centers = _centers(db, query, limit=3)
        if not centers:
            raise ValueError("Procedural index has no usable centers")
        ids = [cluster.id for _, cluster in centers]
        candidates = (
            select(ProceduralSkill, SkillMember.vector)
            .select_from(SkillMember)
            .join(ProceduralSkill, _current_members())
            .where(SkillMember.cluster_id.in_(ids), ProceduralSkill.active)
            .execution_options(populate_existing=True)
        )
        return _rank(db.execute(candidates), query), False
    except Exception:
        logger.warning(
            "Procedural index unavailable; using live flat search", exc_info=True
        )
        return _flat(db, query, use_index=False), True
