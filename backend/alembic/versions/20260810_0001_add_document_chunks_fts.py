from alembic import op

revision = "20260810_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            title_tokens,
            body_tokens,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS document_chunks_fts")
