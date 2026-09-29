"""Metrics and acceptance gates for the intent router evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass

ROUTE_LABELS = ("direct", "simple_rag", "agentic_rag")


@dataclass(frozen=True)
class RoutingMetrics:
    case_count: int
    accuracy: float
    confusion: dict[str, dict[str, int]]


def evaluate_routes(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> RoutingMetrics:
    """Score paired route labels and retain a stable three-way confusion matrix."""

    paired = list(zip(gold, predicted, strict=False))
    confusion = {
        actual: {guess: 0 for guess in ROUTE_LABELS}
        for actual in ROUTE_LABELS
    }
    for actual, guess in paired:
        if actual not in confusion:
            confusion[actual] = {label: 0 for label in ROUTE_LABELS}
        if guess not in confusion[actual]:
            confusion[actual][guess] = 0
        confusion[actual][guess] += 1
    correct = sum(actual == guess for actual, guess in paired)
    accuracy = round(correct / len(paired), 4) if paired else 0.0
    return RoutingMetrics(case_count=len(paired), accuracy=accuracy, confusion=confusion)


def hybrid_meets_gate(
    *,
    hybrid_recall: float,
    bm25_recall: float,
    vector_recall: float,
) -> bool:
    """Require fused retrieval to match or exceed either single-channel baseline."""

    return hybrid_recall >= max(bm25_recall, vector_recall)
