from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from backend.app.core import config
from backend.app.models.memory import ProceduralSkill
from sqlalchemy.orm import Session


def test_v2_database_upgrade_round_trip(tmp_path, monkeypatch):
    url = "sqlite:///" + str(tmp_path / "migration.db")
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
        command.upgrade(cfg, "head")
        with Session(engine) as db:
            assert db.query(ProceduralSkill).count() == 0
            assert db.scalar(text("SELECT content FROM memories")) == "prefer examples"
        command.downgrade(cfg, "20261007_0002")
        assert "procedural_skills" not in inspect(engine).get_table_names()
        with engine.connect() as conn:
            assert (
                conn.scalar(text("SELECT content FROM memories")) == "prefer examples"
            )
    finally:
        engine.dispose()
