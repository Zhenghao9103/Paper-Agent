from backend.app.ingestion.domain import (
    BBox,
    ElementParseStatus,
    ElementType,
    ParsedElement,
    SectionNode,
)
from backend.app.ingestion.references import CrossReferenceResolver, ReferenceParser


def test_reference_parser_canonicalizes_academic_labels() -> None:
    refs = ReferenceParser().parse(
        [
            {
                "uid": "block-1",
                "page_number": 3,
                "text": "As shown in Fig. 2 and Table III, see Eq. (7) and Section 3.2.",
            }
        ]
    )

    assert [(ref.reference_type, ref.normalized_label) for ref in refs] == [
        ("figure", "Figure 2"),
        ("table", "Table III"),
        ("equation", "Eq. 7"),
        ("section", "Section 3.2"),
    ]


def test_cross_reference_resolver_links_elements_and_preserves_unresolved() -> None:
    elements = [
        ParsedElement(
            uid="figure-p1-2",
            element_type=ElementType.FIGURE,
            status=ElementParseStatus.SUCCESS,
            page_number=1,
            bbox=BBox(0, 0, 10, 10),
            metadata={"figure_id": "Figure 2"},
        )
    ]
    refs = ReferenceParser().parse(
        [{"uid": "block-1", "text": "Figure 2 and Table 9 are discussed."}]
    )
    resolved = CrossReferenceResolver().resolve(
        refs, elements, [SectionNode("section-1", "3 Method", 1)]
    )

    assert resolved[0].target_id == "figure-p1-2"
    assert resolved[0].resolution_status == "resolved"
    assert resolved[1].target_id is None
    assert resolved[1].resolution_status == "unresolved"


def test_cross_reference_resolver_keeps_ambiguous_targets_unresolved() -> None:
    elements = [
        ParsedElement(
            uid=uid,
            element_type=ElementType.FIGURE,
            status=ElementParseStatus.SUCCESS,
            page_number=1,
            bbox=BBox(0, 0, 10, 10),
            metadata={"figure_id": "Figure 2"},
        )
        for uid in ("figure-a", "figure-b")
    ]
    refs = ReferenceParser().parse([{"uid": "block-1", "text": "Figure 2"}])

    resolved = CrossReferenceResolver().resolve(refs, elements)

    assert resolved[0].target_id is None
    assert resolved[0].resolution_status == "unresolved"


def test_cross_reference_resolver_links_numbered_section_heading() -> None:
    refs = ReferenceParser().parse([{"uid": "block-1", "text": "See Section 3."}])
    resolved = CrossReferenceResolver().resolve(
        refs, sections=[SectionNode("section-method", "3 Method", 1)]
    )

    assert resolved[0].target_id == "section-method"


def test_cross_reference_resolver_supports_roman_and_mapping_elements() -> None:
    refs = ReferenceParser().parse(
        [{"uid": "block-1", "text": "See Section II and Figure 4."}]
    )
    resolved = CrossReferenceResolver().resolve(
        refs,
        elements=[
            {
                "uid": "figure-4",
                "element_type": "figure",
                "metadata": {"figure_id": "Figure 4"},
            }
        ],
        sections=[SectionNode("section-ii", "II Methods", 1)],
    )

    assert resolved[0].target_id == "section-ii"
    assert resolved[1].target_id == "figure-4"
