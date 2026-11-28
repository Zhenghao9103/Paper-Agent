from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class SkillCluster(Base):
    __tablename__ = "procedural_skill_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Sum of normalized member embeddings. Its direction is the cosine centroid.
    vector_sum: Mapped[list[float]] = mapped_column(JSON, nullable=False)


class SkillMember(Base):
    __tablename__ = "procedural_skill_members"

    skill_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("procedural_skill_clusters.id"),
        nullable=False,
        index=True,
    )
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
