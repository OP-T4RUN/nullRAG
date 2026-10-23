import re
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from app.schemas.internal_models import EvidenceObligation

class SupportRelation(str, Enum):
    DIRECT_SUPPORT = "direct_support"
    PARTIAL_SUPPORT = "partial_support"
    CONTEXTUAL = "contextual"
    CONTRADICTS = "contradicts"
    IRRELEVANT = "irrelevant"

class CandidateEvidence(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    source_text: str
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_rank: Optional[int] = None
    reranker_score: float = 0.0
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_refs: List[str] = Field(default_factory=list)

class SupportAssessment(BaseModel):
    obligation_id: str
    chunk_id: str
    relation: SupportRelation
    support_strength: float  # 0.0 to 1.0
    confidence: float = 1.0
    matched_constraints: List[str] = Field(default_factory=list)
    failed_constraints: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    explanation: str = ""

class EvidenceSupportMatrix:
    """
    EFAR Evidence Support Matrix:
    Assesses what each retrieved candidate contributes to each evidence obligation.
    Enforces deterministic period and scope eligibility before scoring relevance.
    """

    def build_matrix(
        self,
        candidates: List[Dict[str, Any]],
        obligations: List[EvidenceObligation]
    ) -> Dict[str, Dict[str, SupportAssessment]]:
        """
        Returns mapping: obligation_id -> {chunk_id -> SupportAssessment}
        """
        matrix: Dict[str, Dict[str, SupportAssessment]] = {o.obligation_id: {} for o in obligations}

        for obl in obligations:
            for cand in candidates:
                assessment = self._assess_pair(cand, obl)
                matrix[obl.obligation_id][cand["chunk_id"]] = assessment

        return matrix

    def evaluate_candidate_for_obligation(self, cand: Dict[str, Any], obl: EvidenceObligation) -> SupportAssessment:
        """
        Public method to evaluate a single candidate against an obligation.
        """
        return self._assess_pair(cand, obl)

    def _assess_pair(self, cand: Dict[str, Any], obl: EvidenceObligation) -> SupportAssessment:
        chunk_id = cand["chunk_id"]
        text = cand.get("text", "")
        text_lower = text.lower()
        chunk_year = cand.get("fiscal_year")
        chunk_scope = cand.get("scope")
        chunk_entity = cand.get("entity")

        matched_constraints = []
        failed_constraints = []

        # Stage A: Hard constraint checks
        # 1. Period constraint
        required_yr = obl.required_period or obl.hard_constraints.get("fiscal_year")
        if required_yr and chunk_year:
            if required_yr.lower() == chunk_year.lower():
                matched_constraints.append(f"period:{required_yr}")
            else:
                failed_constraints.append(f"period:expected_{required_yr}_got_{chunk_year}")

        # 2. Scope constraint
        required_scope = obl.required_scope or obl.hard_constraints.get("scope")
        if required_scope and chunk_scope and chunk_scope != "unspecified":
            if required_scope.lower() == chunk_scope.lower():
                matched_constraints.append(f"scope:{required_scope}")
            else:
                failed_constraints.append(f"scope:expected_{required_scope}_got_{chunk_scope}")

        # 3. Entity constraint
        required_entity = obl.required_entity or obl.hard_constraints.get("entity")
        if required_entity and chunk_entity:
            if required_entity.lower() in chunk_entity.lower() or chunk_entity.lower() in required_entity.lower():
                matched_constraints.append(f"entity:{required_entity}")
            else:
                # Soft mismatch (e.g. subsidiary name vs parent name)
                pass

        # If hard constraints failed, chunk is ineligible for this specific obligation
        if failed_constraints:
            return SupportAssessment(
                obligation_id=obl.obligation_id,
                chunk_id=chunk_id,
                relation=SupportRelation.IRRELEVANT,
                support_strength=0.0,
                confidence=0.95,
                matched_constraints=matched_constraints,
                failed_constraints=failed_constraints,
                evidence_refs=[f"{cand.get('document_id')}_p{cand.get('page_number')}"],
                explanation=f"Violates hard constraint(s): {', '.join(failed_constraints)}"
            )

        # Stage B: Keyword & Concept Matching
        kw_matches = sum(1 for kw in obl.keywords if kw.lower() in text_lower)
        total_kws = max(1, len(obl.keywords))
        kw_ratio = kw_matches / total_kws

        # Bonus for tables when obligation requires numeric metrics
        table_bonus = 0.20 if cand.get("segment_type") == "table" and "fact" in obl.requirement_type else 0.0

        # Reranker / RRF score influence
        base_relevance = min(1.0, cand.get("rrf_score", 0.0) * 15.0 + cand.get("score", 0.0) * 0.4)

        # Compute support strength (0.0 to 1.0)
        support_strength = min(1.0, (kw_ratio * 0.50) + (base_relevance * 0.35) + table_bonus)

        # Determine SupportRelation
        if kw_matches >= 2 or (kw_matches >= 1 and support_strength >= 0.65):
            relation = SupportRelation.DIRECT_SUPPORT
        elif kw_matches >= 1 or support_strength >= 0.40:
            relation = SupportRelation.PARTIAL_SUPPORT
        elif support_strength >= 0.20:
            relation = SupportRelation.CONTEXTUAL
        else:
            relation = SupportRelation.IRRELEVANT

        return SupportAssessment(
            obligation_id=obl.obligation_id,
            chunk_id=chunk_id,
            relation=relation,
            support_strength=round(support_strength, 3),
            confidence=0.90,
            matched_constraints=matched_constraints,
            failed_constraints=failed_constraints,
            evidence_refs=[f"{cand.get('document_id')}_p{cand.get('page_number')}"],
            explanation=f"Relation '{relation.value}' with support strength {support_strength:.2f} ({kw_matches} keywords matched)"
        )

support_matrix_builder = EvidenceSupportMatrix()
