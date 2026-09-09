import re
from dataclasses import dataclass

# Keeps the terminator attached to the sentence it ends.
SENTENCE_PATTERN = re.compile(r"[^.!?;。！？；]*[.!?;。！？；]+|[^.!?;。！？；]+$")


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.findall(text) if sentence.strip()]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[TextChunk]:
    """Pack whole sentences into chunks of at most `chunk_size` characters.

    Splitting on sentence boundaries instead of raw character offsets keeps each chunk
    a self-contained unit, so retrieved evidence reads as complete statements rather
    than fragments cut mid-clause. Sentences longer than `chunk_size` fall back to an
    overlapping character window.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return []

    contents: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in _split_units(normalized, chunk_size, overlap):
        separator = 1 if current else 0
        if current and current_length + separator + len(unit) > chunk_size:
            contents.append(" ".join(current))
            current = _overlap_units(current, overlap)
            current_length = _joined_length(current)
            if current and current_length + 1 + len(unit) > chunk_size:
                # The carried context plus this sentence would overflow, so start clean.
                current = []
                current_length = 0
            separator = 1 if current else 0
        current.append(unit)
        current_length += separator + len(unit)

    if current:
        contents.append(" ".join(current))
    return [TextChunk(chunk_index=index, content=content) for index, content in enumerate(contents)]


def _split_units(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Return sentences, hard-splitting any that cannot fit in a chunk on their own."""
    units: list[str] = []
    for sentence in split_sentences(text) or [text]:
        if len(sentence) <= chunk_size:
            units.append(sentence)
            continue

        step = max(1, chunk_size - overlap)
        for start in range(0, len(sentence), step):
            piece = sentence[start : start + chunk_size]
            if piece:
                units.append(piece)
            if start + chunk_size >= len(sentence):
                break
    return units


def _overlap_units(units: list[str], overlap: int) -> list[str]:
    """Carry trailing sentences forward as context, up to `overlap` characters."""
    if overlap <= 0:
        return []

    carried: list[str] = []
    total = 0
    for unit in reversed(units):
        addition = len(unit) + (1 if carried else 0)
        if total + addition > overlap:
            break
        carried.insert(0, unit)
        total += addition
    return carried


def _joined_length(units: list[str]) -> int:
    if not units:
        return 0
    return sum(len(unit) for unit in units) + len(units) - 1
