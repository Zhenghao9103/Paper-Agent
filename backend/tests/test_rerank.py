from types import SimpleNamespace

import pytest
import torch
from backend.app.rag import rerank


class FakeBgeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[list[str]]] = []

    def compute_score(self, pairs):
        self.calls.append(pairs)
        return self.scores


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _LengthAwareTokenizer:
    def __call__(
        self,
        values,
        *,
        return_tensors=None,
        add_special_tokens=False,
        max_length,
        truncation,
        **_kwargs,
    ):
        del return_tensors, add_special_tokens, truncation
        encoded = []
        for value in values:
            if value == "query":
                tokens = [1]
            else:
                index_text, length_text = value.removeprefix("passage-").split("-")
                tokens = [int(index_text) + 10] * int(length_text)
            encoded.append(tokens[:max_length])
        return {"input_ids": encoded}

    def prepare_for_model(
        self,
        query_tokens,
        passage_tokens,
        *,
        truncation,
        max_length,
        padding,
    ):
        del truncation, padding
        return {"input_ids": (query_tokens + passage_tokens)[:max_length]}

    def pad(self, values, *, padding, return_tensors, **_kwargs):
        del padding, return_tensors
        width = max(len(value["input_ids"]) for value in values)
        input_ids = [
            value["input_ids"] + [0] * (width - len(value["input_ids"]))
            for value in values
        ]
        return _FakeBatch(input_ids=torch.tensor(input_ids))


class _RecordingModel:
    def __init__(self) -> None:
        self.forward_shapes: list[tuple[int, ...]] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, return_dict):
        assert return_dict is True
        self.forward_shapes.append(tuple(input_ids.shape))
        return SimpleNamespace(logits=input_ids.max(dim=1).values)


class _CpuReranker:
    def __init__(self) -> None:
        self.target_devices = ["cpu"]
        self.query_max_length = None
        self.max_length = 512
        self.tokenizer = _LengthAwareTokenizer()
        self.model = _RecordingModel()

    def compute_score(self, _pairs):
        raise AssertionError("native compute_score must not run for CPU")


def test_cpu_bge_scores_forward_each_candidate_once_in_micro_batches() -> None:
    model = _CpuReranker()
    pairs = [
        ["query", f"passage-{index}-{20 - index}"] for index in range(20)
    ]

    scores = rerank._compute_cpu_bge_scores(model, pairs, batch_size=4)

    assert scores == [float(index + 10) for index in range(20)]
    assert [shape[0] for shape in model.model.forward_shapes] == [4, 4, 4, 4, 4]
    assert sum(shape[0] for shape in model.model.forward_shapes) == 20
    assert [shape[1] for shape in model.model.forward_shapes] == [21, 17, 13, 9, 5]


def test_bge_score_dispatch_uses_single_pass_path_on_cpu() -> None:
    model = _CpuReranker()
    pairs = [["query", "passage-0-5"], ["query", "passage-1-2"]]

    scores = rerank._compute_bge_scores(model, pairs)

    assert scores == [10.0, 11.0]
    assert model.model.forward_shapes == [(2, 6)]


def test_rerank_chunks_uses_bge_reranker_scores(monkeypatch) -> None:
    model = FakeBgeReranker([0.2, 0.95, 0.5])
    monkeypatch.setattr(rerank, "_get_bge_reranker", lambda: model)
    matches = [
        {"content": "general introduction", "metadata": {"chunk_id": 1}, "score": 0.9},
        {"content": "exact answer about the Laplacian", "metadata": {"chunk_id": 2}, "score": 0.4},
        {"content": "related experiment details", "metadata": {"chunk_id": 3}, "score": 0.7},
    ]

    reranked = rerank.rerank_chunks("What uses graph Laplacian?", matches, top_k=2)

    assert model.calls == [
        [
            ["What uses graph Laplacian?", "general introduction"],
            ["What uses graph Laplacian?", "exact answer about the Laplacian"],
            ["What uses graph Laplacian?", "related experiment details"],
        ]
    ]
    assert [match["metadata"]["chunk_id"] for match in reranked] == [2, 3]
    assert reranked[0]["original_score"] == 0.4
    assert reranked[0]["rerank_logit"] == 0.95
    assert reranked[0]["rerank_score"] == pytest.approx(0.72112, abs=1e-5)
    assert reranked[0]["score"] == reranked[0]["rerank_score"]


def test_rerank_chunks_handles_single_float_score(monkeypatch) -> None:
    model = FakeBgeReranker([0.8])
    monkeypatch.setattr(rerank, "_get_bge_reranker", lambda: model)

    reranked = rerank.rerank_chunks(
        "question",
        [{"content": "answer", "metadata": {}, "score": 0.1}],
        top_k=1,
    )

    assert reranked[0]["score"] == pytest.approx(0.68997, abs=1e-5)


def test_rerank_scores_stay_within_unit_range_for_extreme_logits(monkeypatch) -> None:
    """Raw cross-encoder logits are unbounded and often negative.

    They must be squashed into [0, 1] so they are comparable with the cosine
    similarities produced by first-stage recall.
    """
    model = FakeBgeReranker([-9.5, 11.0, -0.4])
    monkeypatch.setattr(rerank, "_get_bge_reranker", lambda: model)
    matches = [
        {"content": "irrelevant boilerplate", "metadata": {"chunk_id": 1}, "score": 0.5},
        {"content": "the exact answer", "metadata": {"chunk_id": 2}, "score": 0.5},
        {"content": "loosely related", "metadata": {"chunk_id": 3}, "score": 0.5},
    ]

    reranked = rerank.rerank_chunks("question", matches, top_k=3)

    assert [match["metadata"]["chunk_id"] for match in reranked] == [2, 3, 1]
    assert all(0.0 <= match["score"] <= 1.0 for match in reranked)
    assert reranked[0]["score"] > 0.99
    assert reranked[-1]["score"] < 0.01


def test_normalize_rerank_score_is_monotonic() -> None:
    scores = [rerank.normalize_rerank_score(value) for value in (-8.0, -1.0, 0.0, 1.0, 8.0)]

    assert scores == sorted(scores)
    assert scores[2] == pytest.approx(0.5)
