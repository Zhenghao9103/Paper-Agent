from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

Language = Literal["zh", "en"]
TaskType = Literal[
    "single_paper",
    "cross_paper_comparison",
    "cross_paper_similarity",
    "insufficient_evidence",
]
Difficulty = Literal["easy", "medium", "hard"]
ScoringMode = Literal["any_evidence", "all_documents", "no_evidence"]


class CorpusPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    file_prefix: str = Field(min_length=1)
    title: str | None = None


class GoldReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str = Field(min_length=1)
    pages: list[PositiveInt] = Field(min_length=1)


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str = Field(min_length=1)
    page: PositiveInt
    text: str = Field(min_length=1)


class GoldVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["draft", "verified"]
    method: Literal["pdf_page_review"]


class RAGGoldCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    language: Language
    task_type: TaskType
    difficulty: Difficulty
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    answer_points: list[str]
    gold: list[GoldReference]
    gold_keywords: list[str]
    evidence: list[GoldEvidence]
    acceptable_variants: list[str]
    required_documents: list[str]
    scoring_mode: ScoringMode
    expect_insufficient_evidence: bool
    unanswerable_reason: str | None
    verification: GoldVerification

    @model_validator(mode="after")
    def validate_case_contract(self) -> RAGGoldCase:
        if self.verification.status != "verified":
            raise ValueError("formal gold cases must be verified")
        if not self.answer_points or any(not point.strip() for point in self.answer_points):
            raise ValueError("answer points must contain non-empty scoring points")
        if len(self.required_documents) != len(set(self.required_documents)):
            raise ValueError("required document keys must be unique")

        gold_documents = {item.doc for item in self.gold}
        evidence_documents = {item.doc for item in self.evidence}
        required_documents = set(self.required_documents)

        if self.expect_insufficient_evidence:
            if self.task_type != "insufficient_evidence":
                raise ValueError("insufficient evidence cases must use the matching task type")
            if self.scoring_mode != "no_evidence":
                raise ValueError("insufficient evidence cases must use no_evidence scoring")
            if self.gold:
                raise ValueError("insufficient evidence cases must not contain gold pages")
            if self.evidence or self.required_documents:
                raise ValueError("insufficient evidence cases must not contain evidence")
            if not self.unanswerable_reason or not self.unanswerable_reason.strip():
                raise ValueError("insufficient evidence cases require an unanswerable reason")
            return self

        if self.task_type == "insufficient_evidence":
            raise ValueError("answerable cases cannot use insufficient_evidence task type")
        if self.unanswerable_reason is not None:
            raise ValueError("answerable cases must not define an unanswerable reason")
        if not self.gold:
            raise ValueError("answerable cases require gold pages")
        if not self.evidence:
            raise ValueError("answerable cases require evidence")
        if not self.gold_keywords:
            raise ValueError("answerable cases require gold keywords")
        if gold_documents != required_documents:
            raise ValueError("gold document keys must match required document keys")
        if evidence_documents != required_documents:
            raise ValueError("evidence documents must match required documents")

        gold_pairs = {
            (reference.doc, page)
            for reference in self.gold
            for page in reference.pages
        }
        if any((item.doc, item.page) not in gold_pairs for item in self.evidence):
            raise ValueError("every evidence page must appear in gold pages")

        document_count = len(required_documents)
        if self.task_type == "single_paper":
            if document_count != 1 or self.scoring_mode != "any_evidence":
                raise ValueError(
                    "single_paper cases require one document and any_evidence scoring"
                )
        else:
            if not 2 <= document_count <= 6:
                raise ValueError("cross-paper cases require between 2 and 6 documents")
            if self.scoring_mode != "all_documents":
                raise ValueError("cross-paper cases require all_documents scoring")
        return self


class RAGGoldDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    corpus: list[CorpusPaper]
    cases: list[RAGGoldCase]

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> RAGGoldDataset:
        corpus_keys = [paper.key for paper in self.corpus]
        if len(corpus_keys) != len(set(corpus_keys)):
            raise ValueError("corpus keys must be unique")
        prefixes = [paper.file_prefix.casefold() for paper in self.corpus]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("corpus file prefixes must be unique")
        if len(self.corpus) != 20:
            raise ValueError("gold dataset must contain exactly 20 corpus papers")
        if len(self.cases) != 100:
            raise ValueError("gold dataset must contain exactly 100 cases")

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case ids must be unique")
        questions = [case.question.strip().casefold() for case in self.cases]
        if len(questions) != len(set(questions)):
            raise ValueError("questions must be unique")

        known_keys = set(corpus_keys)
        for case in self.cases:
            referenced = set(case.required_documents)
            referenced.update(reference.doc for reference in case.gold)
            referenced.update(evidence.doc for evidence in case.evidence)
            unknown = referenced - known_keys
            if unknown:
                raise ValueError(f"unknown document key: {sorted(unknown)}")

        languages = Counter(case.language for case in self.cases)
        if languages != Counter({"zh": 60, "en": 40}):
            raise ValueError("language distribution must be exactly zh=60 and en=40")

        singles = [case for case in self.cases if case.task_type == "single_paper"]
        cross = [
            case
            for case in self.cases
            if case.task_type
            in {"cross_paper_comparison", "cross_paper_similarity"}
        ]
        insufficient = [
            case for case in self.cases if case.task_type == "insufficient_evidence"
        ]
        if (len(singles), len(cross), len(insufficient)) != (60, 30, 10):
            raise ValueError("task distribution must be exactly single=60, cross=30, no-answer=10")

        single_counts = Counter(case.required_documents[0] for case in singles)
        if any(single_counts[key] != 3 for key in corpus_keys):
            raise ValueError("every corpus paper must have exactly three single-paper cases")

        breadth = Counter(len(case.required_documents) for case in cross)
        if sum(breadth[size] for size in (2, 3)) != 18:
            raise ValueError("cross-paper breadth must include 18 cases with 2-3 papers")
        if breadth[4] != 8:
            raise ValueError("cross-paper breadth must include 8 cases with 4 papers")
        if sum(breadth[size] for size in (5, 6)) != 4:
            raise ValueError("cross-paper breadth must include 4 cases with 5-6 papers")

        appearances = Counter(
            document for case in cross for document in case.required_documents
        )
        if any(appearances[key] < 3 for key in corpus_keys):
            raise ValueError("every corpus paper must appear in at least three cross-paper cases")
        return self

    def summary(self) -> dict[str, object]:
        task_counts = Counter(case.task_type for case in self.cases)
        cross_count = sum(
            task_counts[task]
            for task in ("cross_paper_comparison", "cross_paper_similarity")
        )
        return {
            "name": self.name,
            "version": self.version,
            "paper_count": len(self.corpus),
            "case_count": len(self.cases),
            "languages": dict(Counter(case.language for case in self.cases)),
            "task_counts": {
                "single_paper": task_counts["single_paper"],
                "cross_paper": cross_count,
                "insufficient_evidence": task_counts["insufficient_evidence"],
            },
            "verified_count": sum(
                case.verification.status == "verified" for case in self.cases
            ),
        }
