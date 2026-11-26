import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app import models  # noqa: F401 - register all ORM tables


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "fixture.db"))
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()
