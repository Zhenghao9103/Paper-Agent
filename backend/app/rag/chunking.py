from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[TextChunk]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(TextChunk(chunk_index=len(chunks), content=normalized[start:end]))
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks
