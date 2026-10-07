"""Replace importance scores with durable Memory V2 vector synchronization."""

import json

import sqlalchemy as sa

from alembic import op

revision = "20261007_0002"
down_revision = "20260810_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    tables = set(sa.inspect(connection).get_table_names())
    rows = []
    if "memories" in tables:
        rows = (
            connection.execute(sa.text("SELECT id, content, status FROM memories ORDER BY id"))
            .mappings()
            .all()
        )
    pending_memory_ids = set()
    existing_memory_ids = {row["id"] for row in rows}
    pending_payload_updates = []
    if "memory_sync_operations" in tables:
        pending_operations = connection.execute(
            sa.text("SELECT id, payload FROM memory_sync_operations WHERE status = 'pending'")
        ).mappings()
        for operation in pending_operations:
            payload = json.loads(operation["payload"])
            pending_memory_ids.update(row["memory_id"] for row in payload)
            changed = False
            for row in payload:
                if (
                    row["memory_id"] in existing_memory_ids
                    and row.get("replace_metadata") is not True
                ):
                    row["replace_metadata"] = True
                    changed = True
            if changed:
                pending_payload_updates.append(
                    {
                        "operation_id": operation["id"],
                        "payload": json.dumps(payload, ensure_ascii=False),
                    }
                )
    orphan_ids = [
        row["id"]
        for row in rows
        if row["status"] == "pending_sync" and row["id"] not in pending_memory_ids
    ]
    if orphan_ids:
        raise RuntimeError(
            f"Memory V2 migration cannot continue: pending_sync memories {orphan_ids} "
            "have no pending sync operation. Repair their sync operation or restore "
            "their intended final status before rerunning this migration."
        )

    if "memories" not in tables:
        # The preceding revision only creates the FTS table. Bootstrap a fresh
        # database without depending on the application's mutable ORM models.
        op.create_table(
            "memories",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("memory_type", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("source_type", sa.Text(), nullable=False),
            sa.Column("source_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("is_pinned", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_memories_id", "memories", ["id"])
        op.create_index("ix_memories_status", "memories", ["status"])
    elif "importance_score" in {
        column["name"] for column in sa.inspect(connection).get_columns("memories")
    }:
        with op.batch_alter_table("memories") as batch_op:
            batch_op.drop_column("importance_score")

    if "memory_sync_operations" not in tables:
        op.create_table(
            "memory_sync_operations",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("action", sa.Text(), nullable=False),
            sa.Column("payload", sa.Text(), nullable=False),
            sa.Column("source_session_id", sa.Integer(), nullable=True),
            sa.Column("source_checkpoint_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index("ix_memory_sync_operations_id", "memory_sync_operations", ["id"])
        op.create_index("ix_memory_sync_operations_status", "memory_sync_operations", ["status"])

    if pending_payload_updates:
        # Preserve existing operations and timestamps while clearing legacy vector
        # metadata through their existing replay, without a conflicting REINDEX.
        connection.execute(
            sa.text(
                "UPDATE memory_sync_operations SET payload = :payload WHERE id = :operation_id"
            ),
            pending_payload_updates,
        )
    operations = sa.Table("memory_sync_operations", sa.MetaData(), autoload_with=connection)
    for row in rows:
        if row["id"] in pending_memory_ids:
            continue
        connection.execute(
            operations.insert().values(
                action="REINDEX",
                payload=json.dumps(
                    [
                        {
                            "memory_id": row["id"],
                            "content": row["content"],
                            "status": row["status"],
                            "refresh_updated_at": False,
                            "replace_metadata": True,
                        }
                    ],
                    ensure_ascii=False,
                ),
            )
        )
        # Raw SQL deliberately bypasses the ORM's updated_at on-update behavior.
        connection.execute(
            sa.text("UPDATE memories SET status = 'pending_sync' WHERE id = :memory_id"),
            {"memory_id": row["id"]},
        )


def downgrade() -> None:
    connection = op.get_bind()
    tables = set(sa.inspect(connection).get_table_names())
    if "memory_sync_operations" in tables:
        pending_actions = (
            connection.execute(
                sa.text(
                    "SELECT DISTINCT action FROM memory_sync_operations "
                    "WHERE status = 'pending' AND action != 'REINDEX' ORDER BY action"
                )
            )
            .scalars()
            .all()
        )
        if pending_actions:
            raise RuntimeError(
                "Memory V2 downgrade cannot continue with pending sync operations: "
                f"{', '.join(pending_actions)}. Finish memory sync before retrying "
                "downgrade to preserve durable work."
            )
        if "memories" in tables:
            payloads = (
                connection.execute(
                    sa.text(
                        "SELECT payload FROM memory_sync_operations "
                        "WHERE action = 'REINDEX' AND status = 'pending' ORDER BY id"
                    )
                )
                .scalars()
                .all()
            )
            for payload in payloads:
                for row in json.loads(payload):
                    connection.execute(
                        sa.text(
                            "UPDATE memories SET status = :status "
                            "WHERE id = :memory_id AND status = 'pending_sync'"
                        ),
                        {"status": row["status"], "memory_id": row["memory_id"]},
                    )
        op.drop_table("memory_sync_operations")
    if "memories" in tables and "importance_score" not in {
        column["name"] for column in sa.inspect(connection).get_columns("memories")
    }:
        with op.batch_alter_table("memories") as batch_op:
            # Scores were deliberately removed; downgrade can only restore the
            # old schema's neutral default, not recover historical scores.
            batch_op.add_column(
                sa.Column("importance_score", sa.Float(), nullable=False, server_default="0.5")
            )
