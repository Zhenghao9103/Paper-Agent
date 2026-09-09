from ..db.session import SessionLocal, engine
from .bm25_store import ensure_bm25_schema, rebuild_bm25_index


def main() -> None:
    ensure_bm25_schema(engine)
    db = SessionLocal()
    try:
        count = rebuild_bm25_index(db)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    print(f"BM25 index rebuilt: {count} chunks")


if __name__ == "__main__":
    main()
