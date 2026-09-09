from backend.app.services.arxiv_search import build_arxiv_query


def test_build_arxiv_query_searches_plain_keywords_across_all_fields() -> None:
    assert (
        build_arxiv_query("multimodal rag agent")
        == 'all:"multimodal" AND all:"rag" AND all:"agent"'
    )


def test_build_arxiv_query_preserves_advanced_arxiv_syntax() -> None:
    assert build_arxiv_query('ti:"retrieval augmented generation" AND cat:cs.CL') == (
        'ti:"retrieval augmented generation" AND cat:cs.CL'
    )
