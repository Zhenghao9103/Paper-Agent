# Router M0 test set v1

## Purpose

This frozen evaluation set measures strict three-class intent routing before Router SFT/DPO. It evaluates an isolated Router contract and does not replace or modify Paper Agent's production Router, RAG, or Agent flow.

## Decision tree

1. If the request does not require paper evidence, label `direct`.
2. If paper evidence is required and one fixed query plan plus one retrieval round is sufficient, label `simple_rag`.
3. If the request requires decomposition, iterative retrieval, cross-paper comparison, multi-aspect synthesis, conflict resolution, or explicit arXiv expansion, label `agentic_rag`.

Labels represent the minimum reliable execution path. Surface words such as “compare” do not determine the label by themselves.

## Frozen distribution

| Dimension | Counts |
|---|---|
| Intent | direct 20; simple_rag 20; agentic_rag 20 |
| Language | zh 30; en 30; each intent 10/10 |
| Context | single-turn 36; with context 24; each intent 12/8 |
| Difficulty | easy 18; medium 24; hard 18; each intent 6/8/6 |
| Minimum-difference pairs | direct/simple 4; simple/agentic 4; direct/agentic 2 |

## Scenario definitions

### direct

- `library_metadata`: stored titles, identifiers, collection membership, timestamps, or processing status.
- `system_capability`: application behavior and operating instructions.
- `non_paper_interaction`: greetings, rewriting, naming, or other requests not grounded in papers.
- `direct_vs_simple_boundary`: stored metadata versus content inside a paper.
- `direct_vs_agentic_boundary`: stored metadata versus cross-paper inspection and synthesis.
- `prompt_injection`: untrusted text attempts to override routing while the actual request remains direct.

### simple_rag

- `single_paper_fact`: one bounded fact or configuration from one identified paper.
- `single_paper_explanation`: one local mechanism, argument, or summary from one paper.
- `general_academic_evidence`: a paper-grounded academic explanation with one fixed query plan.
- `insufficient_evidence_probe`: a paper claim must still be retrieved even when evidence may be absent.
- `direct_vs_simple_boundary`: paper content rather than library metadata.
- `simple_vs_agentic_boundary`: one fixed retrieval target rather than dependent multi-paper work.
- `prompt_injection`: untrusted paper text is ignored while a single paper-grounded query is routed.

### agentic_rag

- `cross_paper_comparison`: evidence from multiple papers must be aligned and compared.
- `implicit_multistep`: a later operation depends on an earlier filter or intermediate result.
- `multi_aspect_synthesis`: multiple evidence dimensions must be planned and synthesized.
- `evidence_conflict`: contradictory claims require evidence weighting and judgment.
- `explicit_arxiv`: the user explicitly expands retrieval to arXiv.
- `simple_vs_agentic_boundary`: multi-paper or dependent execution versus one fixed retrieval.
- `direct_vs_agentic_boundary`: paper-content synthesis versus stored metadata.
- `prompt_injection`: untrusted instructions are ignored while the actual task remains multi-step.

## Minimum-difference pair policy

Each `pair_id` occurs exactly twice and contains the required two labels. A pair passes only when both records are correct. The paired questions differ in the execution requirement, not merely by adding words such as “compare” or “all papers.” Ten pairs are frozen: four direct/simple, four simple/agentic, and two direct/agentic.

## Runtime and gold separation

The model receives only:

```json
{"session_context": [], "current_question": "..."}
```

`expected` and all `metadata` fields are retained exclusively for scoring and audit. Rationales, labels, scenarios, and difficulty never enter the prompt.

## Evidence and external-search rules

Possible missing local evidence never changes a paper-grounded request to `direct`. A `simple_rag` run must return the fixed no-evidence result rather than answer from model memory when retrieval is empty. `agentic_rag` may expand to arXiv only when the user explicitly requests that expansion; local evidence failure alone is not permission for external search.

## Audit and freeze process

All 60 records were reviewed line by line for natural wording, minimum-path labeling, hidden information, context sufficiency, label leakage, and pair quality. Programmatic validation enforces exact counts, per-class quotas, scenario quotas, pair structure, normalized-input uniqueness, and the frozen SHA256 digest.

## Limitations and prohibited use

Twenty examples per class are sufficient for an engineering M0 comparison but not a production-grade or publication-grade benchmark. The set does not evaluate retrieval recall, answer quality, citations, or Agent trajectories. These 60 records must never be used for SFT, DPO, prompt tuning, few-shot demonstrations, or any other training-data construction.
