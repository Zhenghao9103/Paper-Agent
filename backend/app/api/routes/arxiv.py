from fastapi import APIRouter, Query

from ...schemas.arxiv import ArxivPaper
from ...services.arxiv_search import search_arxiv

router = APIRouter(prefix="/arxiv", tags=["arxiv"])


@router.get("/search", response_model=list[ArxivPaper])
def search(
    query: str = Query(..., min_length=1),
    max_results: int = Query(default=5, ge=1, le=20),
) -> list[ArxivPaper]:
    return search_arxiv(query, max_results=max_results)
