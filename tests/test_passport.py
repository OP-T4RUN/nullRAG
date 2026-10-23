from app.ingestion.passport import PassportExtractor
from app.ingestion.deduplication import DuplicateDetector
from app.schemas.internal_models import ParsedSegment

def test_passport_extraction_and_dedup():
    extractor = PassportExtractor()
    detector = DuplicateDetector()

    segments_doc_a = [
        ParsedSegment(
            segment_id="docA_p1_text",
            document_id="docA",
            page_number=1,
            segment_type="text",
            text="Acme Corp Annual Report. Consolidated Financial Statements for FY2024. Total credit exposure was $150 million.",
            content_hash="hash_a1"
        )
    ]

    segments_doc_b = [
        ParsedSegment(
            segment_id="docB_p1_text",
            document_id="docB",
            page_number=1,
            segment_type="text",
            text="Acme Corp Annual Report. Consolidated Financial Statements for FY2025. Total credit exposure was $185 million.",
            content_hash="hash_b1"
        )
    ]

    passport_a = extractor.extract_passport("test_col", "docA", "acme_fy2024.pdf", segments_doc_a)
    passport_b = extractor.extract_passport("test_col", "docB", "acme_fy2025.pdf", segments_doc_b)

    assert passport_a.fiscal_year == "FY2024"
    assert passport_b.fiscal_year == "FY2025"
    assert passport_a.scope == "consolidated"

    detector.register_passport(passport_a)
    rel_type, rel_id = detector.check_relationship(passport_b)

    # They should NOT be treated as exact duplicates even if highly similar text, because fiscal years differ
    assert rel_type != "exact_duplicate"
