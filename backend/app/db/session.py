from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from ..core.paths import BACKEND_ROOT


def normalize_database_url(database_url: str) -> str:
    if not database_url.startswith("sqlite:///./"):
        return database_url

    relative_path = database_url.removeprefix("sqlite:///./")
    absolute_path = (BACKEND_ROOT / relative_path).resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{absolute_path.as_posix()}"


engine = create_engine(
    normalize_database_url(get_settings().database_url),
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
