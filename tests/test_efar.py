import pytest
from app.knowledge.models import MetadataAssertion, EvidenceRelationship, AssertionStatus, RelationshipType
from app.knowledge.registry import ConceptRegistry
from app.knowledge.assertions import AssertionStore
from app.schemas.internal_models import EvidenceObligation
from app.retrieval.support_matrix import EvidenceSupportMatrix
from app.retrieval.selector import CoverageConstrainedSelector
from app.retrieval.consistency import ConsistencyAnalyzer
from app.generation.validation import ClaimValidator

def test_concept_registry_open_world():
    registry = ConceptRegistry()
    registry.register_concept(
        raw_attribute_name="allowance_for_credit_losses",
        canonical_concept="credit_loss_provision"
    )
    # Check concept resolution
    concept, conf = registry.canonicalize("allowance_for_credit_losses")
    assert concept == "credit_loss_provision"
    assert conf >= 0.85
    
    # Check built-in concept resolution
    c_builtin, c_conf = registry.canonicalize("credit risk exposure")
    assert c_builtin == "credit_risk_exposure"
    
    # Check unseen attribute returns normalized slug with default confidence without error
    unseen_concept, u_conf = registry.canonicalize("completely_unseen_esg_metric_x")
    assert unseen_concept == "completely_unseen_esg_metric_x"
    assert u_conf == 0.70

def test_assertion_store_crud():
    store = AssertionStore()
    
    assertion = MetadataAssertion(
        assertion_id="ast_test_001",
        document_id="doc_gammon_2025",
        raw_attribute_name="total_liabilities",
        canonical_concept="liabilities",
        value=7450.5,
        value_type="float",
        unit="INR Crores",
        evidence_refs=["doc_gammon_2025_p42_c1"],
        confidence=0.95,
        status=AssertionStatus.SUPPORTED
    )
    
    store.save_assertion(assertion)
    retrieved = store.get_assertions_for_document("doc_gammon_2025")
    matching = [a for a in retrieved if a.assertion_id == "ast_test_001"]
    assert len(matching) == 1
    assert matching[0].raw_attribute_name == "total_liabilities"
    assert matching[0].value == 7450.5
    assert matching[0].unit == "INR Crores"

def test_evidence_support_matrix_hard_constraint():
    matrix = EvidenceSupportMatrix()
    
    # Obligation strictly requires FY2025
    obligation = EvidenceObligation(
        obligation_id="obl_fy25",
        description="Consolidated revenue for FY2025",
        requirement_type="numeric_fact",
        required_period="FY2025",
        hard_constraints={"fiscal_year": "FY2025"},
        keywords=["revenue", "sales"]
    )
    
    # Candidate 1: Belonging to FY2024
    cand_fy24 = {
        "chunk_id": "c_2024",
        "document_id": "doc_2024",
        "page_number": 10,
        "fiscal_year": "FY2024",
        "text": "Total consolidated revenue was 1,200 million in FY2024."
    }
    
    # Candidate 2: Belonging to FY2025
    cand_fy25 = {
        "chunk_id": "c_2025",
        "document_id": "doc_2025",
        "page_number": 12,
        "fiscal_year": "FY2025",
        "text": "Total consolidated revenue reached 1,500 million in FY2025."
    }
    
    assessment_fy24 = matrix.evaluate_candidate_for_obligation(cand_fy24, obligation)
    assessment_fy25 = matrix.evaluate_candidate_for_obligation(cand_fy25, obligation)
    
    # FY2024 candidate must fail hard constraint
    assert assessment_fy24.relation.value == "irrelevant"
    assert any("period" in fc for fc in assessment_fy24.failed_constraints)
    assert assessment_fy24.support_strength == 0.0
    
    # FY2025 candidate satisfies constraints
    assert assessment_fy25.relation.value in ["direct_support", "partial_support"]
    assert assessment_fy25.support_strength > 0.0

def test_coverage_constrained_selector():
    matrix_builder = EvidenceSupportMatrix()
    selector = CoverageConstrainedSelector()
    
    # Two distinct obligations
    obl_rev = EvidenceObligation(
        obligation_id="obl_revenue",
        description="Revenue in FY2025",
        keywords=["revenue", "sales"],
        weight=1.0
    )
    obl_exp = EvidenceObligation(
        obligation_id="obl_expenses",
        description="Operational expenses in FY2025",
        keywords=["expenses", "overhead"],
        weight=1.0
    )
    obligations = [obl_rev, obl_exp]
    
    # Candidate A supports revenue
    cand_a = {
        "chunk_id": "c_a",
        "document_id": "doc_1",
        "page_number": 5,
        "text": "Revenue for the period was $500M with sales expanding across sectors."
    }
    # Candidate B is redundant with A (same topic, same page)
    cand_b = {
        "chunk_id": "c_b",
        "document_id": "doc_1",
        "page_number": 5,
        "text": "Our sales revenue reached $500M representing strong performance."
    }
    # Candidate C supports expenses (complementary)
    cand_c = {
        "chunk_id": "c_c",
        "document_id": "doc_1",
        "page_number": 10,
        "text": "Operational expenses were maintained at $320M, reducing overhead."
    }
    candidates = [cand_a, cand_b, cand_c]
    
    support_matrix = matrix_builder.build_matrix(candidates, obligations)
    selected_chunks, updated_obls, summary = selector.select_evidence(
        candidates=candidates,
        obligations=obligations,
        support_matrix=support_matrix,
        max_context_chunks=2
    )
    
    selected_ids = [c["chunk_id"] for c in selected_chunks]
    # Both distinct obligations should be covered (cand_a and cand_c selected)
    assert "c_a" in selected_ids
    assert "c_c" in selected_ids
    assert summary["obligations_supported"] >= 1

def test_consistency_analyzer():
    analyzer = ConsistencyAnalyzer()
    
    # Two chunks with different scopes: Consolidated vs Standalone
    chunk_consolidated = {
        "chunk_id": "c_cons",
        "document_id": "doc_gammon",
        "scope": "consolidated",
        "text": "Consolidated net loss after tax was INR 540 Crores."
    }
    chunk_standalone = {
        "chunk_id": "c_stand",
        "document_id": "doc_gammon",
        "scope": "standalone",
        "text": "Standalone net loss after tax was INR 120 Crores."
    }
    
    report = analyzer.analyze_consistency([chunk_consolidated, chunk_standalone])
    # Should detect scope difference and explain it rather than declaring a silent error
    assert "scope_differentiations" in report
    assert len(report["scope_differentiations"]) > 0
    assert report["scope_differentiations"][0]["type"] == "scope_distinction"

def test_claim_validator():
    validator = ClaimValidator()
    
    selected_chunks = [
        {"chunk_id": "c_1", "document_id": "Gammon_AR_2025", "page_number": 42},
        {"chunk_id": "c_2", "document_id": "Gammon_AR_2025", "page_number": 88}
    ]
    obligations = [
        EvidenceObligation(obligation_id="o1", description="Going concern", status="supported")
    ]
    
    valid_text = "The company reported going concern uncertainty [Gammon_AR_2025, Page 42] and debt restructuring [Gammon_AR_2025, Page 88]."
    hallucinated_text = "The revenue grew by 150% [UnknownDoc, Page 999]."
    
    res_valid = validator.validate_answer(valid_text, selected_chunks, obligations)
    assert res_valid["is_valid"] is True
    assert res_valid["citations_verified"] == 2
    assert len(res_valid["hallucinated_citations"]) == 0
    
    res_invalid = validator.validate_answer(hallucinated_text, selected_chunks, obligations)
    assert res_invalid["is_valid"] is False
    assert len(res_invalid["hallucinated_citations"]) > 0
