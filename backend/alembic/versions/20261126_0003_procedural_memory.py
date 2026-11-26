"""Add instance-local procedural memory and its derived index."""

import sqlalchemy as sa
from alembic import op

revision = "20261126_0003"
down_revision = "20261007_0002"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    column = sa.Column
    sa.Table(
        "procedural_skills",
        metadata,
        column("id", sa.String, primary_key=True),
        column("version", sa.Integer, nullable=False),
        column("active", sa.Boolean, nullable=False),
        column("condition", sa.Text, nullable=False),
        column("steps", sa.JSON, nullable=False),
        column("constraints", sa.JSON, nullable=False),
        column("source_event_id", sa.Integer, nullable=False),
        column("source_session_id", sa.Integer, nullable=False),
    )
    sa.Table(
        "procedural_feedback",
        metadata,
        column("message_id", sa.Integer, primary_key=True, autoincrement=False),
        column("status", sa.String, index=True, nullable=False),
        column("reason", sa.String),
        column("trace_id", sa.Integer),
        column("strategy", sa.JSON),
        column("attempts", sa.Integer, nullable=False),
        column("skill_id", sa.String),
        column("version", sa.Integer),
        column("action", sa.String),
    )
    sa.Table(
        "procedural_sync",
        metadata,
        column("id", sa.Integer, primary_key=True),
        column("skill_id", sa.String, nullable=False),
        column("version", sa.Integer, nullable=False),
        column("attempts", sa.Integer, nullable=False),
        column("status", sa.String, nullable=False, index=True),
        column("last_error", sa.String),
    )
    sa.Table(
        "procedural_skill_clusters",
        metadata,
        column("id", sa.Integer, primary_key=True),
        column("size", sa.Integer, nullable=False),
        column("vector_sum", sa.JSON, nullable=False),
    )
    sa.Table(
        "procedural_skill_members",
        metadata,
        column("skill_id", sa.String, primary_key=True),
        column("version", sa.Integer, nullable=False),
        column("vector", sa.JSON, nullable=False),
        column(
            "cluster_id",
            sa.Integer,
            sa.ForeignKey("procedural_skill_clusters.id"),
            index=True,
            nullable=False,
        ),
    )
    metadata.create_all(op.get_bind())


def downgrade():
    for table in (
        "procedural_skill_members",
        "procedural_skill_clusters",
        "procedural_sync",
        "procedural_feedback",
        "procedural_skills",
    ):
        op.drop_table(table)
