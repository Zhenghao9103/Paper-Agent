import re

import arxiv

from ..schemas.arxiv import ArxivPaper

ADVANCED_QUERY_RE = re.compile(r"\b(?:all|ti|au|abs|co|jr|cat|id):|\b(?:AND|OR|ANDNOT)\b")


def build_arxiv_query(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        return cleaned
    if ADVANCED_QUERY_RE.search(cleaned):
        return cleaned

    keywords = re.findall(r"[\w.-]+", cleaned)
    if not keywords:
        return cleaned
    return " AND ".join(f'all:"{keyword}"' for keyword in keywords)


def search_arxiv(query: str, max_results: int = 5) -> list[ArxivPaper]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=build_arxiv_query(query),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers: list[ArxivPaper] = []
    for result in client.results(search):
        papers.append(
            ArxivPaper(
                title=result.title,
                authors=[author.name for author in result.authors],
                summary=result.summary,
                published=result.published.date().isoformat(),
                pdf_url=result.pdf_url,
                entry_url=result.entry_id,
            )
        )
    return papers
