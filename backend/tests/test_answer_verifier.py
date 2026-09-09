import json

from backend.app.agents import answer_verifier
from backend.app.schemas.evidence import (
    EvidencePack,
    EvidencePackItem,
    PackedLocalSource,
    ResearchClaim,
)
from backend.app.services.model_clients import ModelClientError


def _pack() -> EvidencePack:
    return EvidencePack(
        claims=[
            ResearchClaim(
                claim_id="C1",
                question="What role does OT play?",
                status="covered",
                evidence_ids=["E1"],
            ),
            ResearchClaim(
                claim_id="C2",
                question="Is there an ablation?",
                status="covered",
                evidence_ids=["E2"],
            ),
        ],
        items=[
            EvidencePackItem(
                evidence_id="E1",
                statement="OT creates balanced assignments.",
                supports_claim_ids=["C1"],
                confidence=0.9,
                sources=[
                    PackedLocalSource(
                        chunk_id=21,
                        document_id=7,
                        title="OT Paper",
                        page_number=3,
                        chunk_index=1,
                        excerpt="OT creates balanced assignments.",
                    )
                ],
            ),
            EvidencePackItem(
                evidence_id="E2",
                statement="Removing the objective lowers ACC.",
                supports_claim_ids=["C2"],
                confidence=0.9,
                sources=[
                    PackedLocalSource(
                        chunk_id=71,
                        document_id=7,
                        title="OT Paper",
                        page_number=8,
                        chunk_index=2,
                        excerpt="The ablation lowers ACC.",
                    )
                ],
            ),
        ],
        token_count=100,
    )


def test_verifier_reports_unsupported_and_missing_claims() -> None:
    payload = {
        "passed": False,
        "claims": [
            {
                "claim_id": "C1",
                "status": "unsupported",
                "evidence_ids": ["E1"],
                "reason": "The answer overstates E1.",
            },
            {
                "claim_id": "C2",
                "status": "missing",
                "evidence_ids": [],
                "reason": "The answer omits the ablation.",
            },
        ],
        "unsupported_spans": ["always prevents collapse"],
        "citation_errors": [],
        "repair_instruction": "Remove the absolute claim and add the ablation.",
    }

    result = answer_verifier.verify_answer(
        "Why OT?",
        "OT always prevents collapse.",
        ["E1"],
        _pack(),
        verify_callable=lambda messages: payload,
    )

    assert result.passed is False
    assert [claim.status for claim in result.claims] == ["unsupported", "missing"]


def test_verifier_receives_complete_output_schema() -> None:
    captured = []
    payload = {
        "passed": True,
        "claims": [
            {
                "claim_id": "C1",
                "status": "supported",
                "evidence_ids": ["E1"],
                "reason": "Supported.",
            },
            {
                "claim_id": "C2",
                "status": "supported",
                "evidence_ids": ["E2"],
                "reason": "Supported.",
            },
        ],
        "unsupported_spans": [],
        "citation_errors": [],
        "repair_instruction": "",
    }

    def verify(messages):
        captured.append(messages)
        return payload

    result = answer_verifier.verify_answer(
        "Why OT?",
        "Grounded answer.",
        ["E1", "E2"],
        _pack(),
        verify_callable=verify,
    )

    request = json.loads(captured[0][1]["content"])
    assert request["output_schema"] == answer_verifier.AnswerVerification.model_json_schema()
    assert result.passed is True


def test_verifier_schema_error_reports_safe_diagnostics() -> None:
    diagnostics = {}

    result = answer_verifier.verify_answer(
        "Why OT?",
        "Grounded answer.",
        ["E1", "E2"],
        _pack(),
        verify_callable=lambda messages: {
            "passed": True,
            "claims": [{"claim_id": "C1"}],
        },
        diagnostics=diagnostics,
    )

    assert result.passed is False
    assert diagnostics["error_category"] == "verifier_schema_error"
    assert diagnostics["provider_category"] is None
    assert diagnostics["passed"] is False
    assert diagnostics["validation_errors"]
    assert len(diagnostics["validation_errors"]) <= 8
    assert all(
        set(item) == {"loc", "type"} for item in diagnostics["validation_errors"]
    )
    assert any(
        item["loc"] == "claims.0.status"
        for item in diagnostics["validation_errors"]
    )
    assert "Grounded answer" not in str(diagnostics)


def test_verifier_provider_error_reports_category_without_message() -> None:
    diagnostics = {}

    def unavailable(messages):
        raise ModelClientError(
            "provider message must not enter trace",
            category="timeout",
            attempts=2,
        )

    result = answer_verifier.verify_answer(
        "Why OT?",
        "Grounded answer.",
        ["E1", "E2"],
        _pack(),
        verify_callable=unavailable,
        diagnostics=diagnostics,
    )

    assert result.passed is False
    assert diagnostics["error_category"] == "verifier_model_error"
    assert diagnostics["provider_category"] == "timeout"
    assert diagnostics["provider_attempts"] == 2
    assert diagnostics["passed"] is False
    assert "provider message" not in str(diagnostics)


def test_verifier_rejects_unknown_evidence_ids() -> None:
    payload = {
        "passed": True,
        "claims": [
            {
                "claim_id": "C1",
                "status": "supported",
                "evidence_ids": ["E99"],
                "reason": "Invented reference.",
            },
            {
                "claim_id": "C2",
                "status": "supported",
                "evidence_ids": ["E2"],
                "reason": "Supported.",
            },
        ],
        "unsupported_spans": [],
        "citation_errors": [],
        "repair_instruction": "",
    }

    result = answer_verifier.verify_answer(
        "Why OT?",
        "Answer",
        ["E1", "E2"],
        _pack(),
        verify_callable=lambda messages: payload,
    )

    assert result.passed is False
    assert "unknown_evidence_id" in result.citation_errors


def test_verifier_requires_every_claim_assessment() -> None:
    payload = {
        "passed": True,
        "claims": [
            {
                "claim_id": "C1",
                "status": "supported",
                "evidence_ids": ["E1"],
                "reason": "Supported.",
            }
        ],
        "unsupported_spans": [],
        "citation_errors": [],
        "repair_instruction": "",
    }

    result = answer_verifier.verify_answer(
        "Why OT?",
        "Answer",
        ["E1", "E2"],
        _pack(),
        verify_callable=lambda messages: payload,
    )

    assert result.passed is False
    assert "incomplete_claim_assessment" in result.citation_errors
