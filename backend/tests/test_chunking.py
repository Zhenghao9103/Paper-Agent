from backend.app.rag.chunking import chunk_text, split_sentences

SENTENCE_TERMINATORS = ".!?;。！？；"


def test_chunk_text_returns_overlapping_chunks() -> None:
    chunks = chunk_text("word " * 300, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert len(chunks[0].content) <= 100


def test_chunk_text_never_exceeds_chunk_size() -> None:
    text = " ".join(f"Sentence number {index} explains one idea." for index in range(60))

    chunks = chunk_text(text, chunk_size=120, overlap=30)

    assert chunks
    assert all(len(chunk.content) <= 120 for chunk in chunks)


def test_chunk_text_ends_chunks_at_sentence_boundaries() -> None:
    text = (
        "Spectral clustering builds a graph Laplacian. "
        "The method then computes eigenvectors. "
        "Finally it runs k-means on the embedding."
    )

    chunks = chunk_text(text, chunk_size=80, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.content[-1] in SENTENCE_TERMINATORS for chunk in chunks)


def test_chunk_text_carries_previous_sentence_as_overlap() -> None:
    chunks = chunk_text("Alpha one. Beta two. Gamma three. Delta four.", chunk_size=25, overlap=12)

    contents = [chunk.content for chunk in chunks]

    assert contents == [
        "Alpha one. Beta two.",
        "Beta two. Gamma three.",
        "Gamma three. Delta four.",
    ]


def test_chunk_text_splits_chinese_sentences() -> None:
    text = "本文提出了一种谱聚类方法。该方法先构建相似度图。最后在嵌入空间上聚类。"

    chunks = chunk_text(text, chunk_size=20, overlap=0)

    assert len(chunks) == 3
    assert all(chunk.content[-1] in SENTENCE_TERMINATORS for chunk in chunks)


def test_chunk_text_returns_nothing_for_blank_text() -> None:
    assert chunk_text("   \n\t  ") == []


def test_split_sentences_keeps_terminators() -> None:
    assert split_sentences("One thing. Another thing! A third?") == [
        "One thing.",
        "Another thing!",
        "A third?",
    ]
