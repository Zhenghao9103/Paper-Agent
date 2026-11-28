from pathlib import Path
from shutil import copyfile

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from backend.app.core import config
from backend.app.models.memory import Memory, ProceduralSkill
from backend.app.services import memory
from sqlalchemy.orm import Session


def test_v2_database_upgrade_round_trip(tmp_path, monkeypatch, semantic_index):
    original = tmp_path / "v2.db"
    url = "sqlite:///" + str(original)
    monkeypatch.setattr(config.get_settings(), "database_url", url)
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "backend/alembic.ini"))
    command.upgrade(cfg, "20261007_0002")
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO memories (memory_type,content,source_type,status,is_pinned,created_at,updated_at) "
                    "VALUES ('user_preference','prefer examples','chat','active',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
                )
            )
        with Session(engine) as db:
            memory.upsert_memory_vector(db.query(Memory).one())
        engine.dispose()
        copy = tmp_path / "upgrade.db"
        copyfile(original, copy)
        url = "sqlite:///" + str(copy)
        monkeypatch.setattr(config.get_settings(), "database_url", url)
        engine = create_engine(url)
        command.upgrade(cfg, "head")
        with Session(engine) as db:
            assert db.query(ProceduralSkill).count() == 0
            assert db.scalar(text("SELECT content FROM memories")) == "prefer examples"
            hits = memory.query_relevant_memories(db, "how should I explain this?")
            assert [hit.content for hit in hits] == ["prefer examples"]
        command.downgrade(cfg, "20261007_0002")
        assert "procedural_skills" not in inspect(engine).get_table_names()
        with engine.connect() as conn:
            assert (
                conn.scalar(text("SELECT content FROM memories")) == "prefer examples"
            )
    finally:
        engine.dispose()
