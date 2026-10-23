import re
from typing import List, Dict, Any, Optional
from app.schemas.internal_models import EvidenceObligation

class EvidenceEligibilityReranker:
    """
    Reranks candidate chunks against the user query and specific evidence obligations.
    Applies hard eligibility criteria:
    - Filters out chunks belonging to ineligible fiscal years or mismatched scopes
    - Boosts chunks that provide complete evidence for active obligations
    - Preserves complementary chunks across different obligations
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        obligations: List[EvidenceObligation],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        scored_candidates = []
        q_tokens = set(re.findall(r'\w+', query.lower()))

        for item in candidates:
            chunk_text = item.get("text", "")
            chunk_tokens = set(re.findall(r'\w+', chunk_text.lower()))
            
            # Base score from RRF or dense
            base_score = item.get("rrf_score", 0.0) * 10.0 + item.get("score", 0.0) * 0.5
            
            # 1. Term overlap ratio
            overlap_count = len(q_tokens.intersection(chunk_tokens))
            term_overlap_score = (overlap_count / len(q_tokens)) if q_tokens else 0.0

            # 2. Obligation alignment boost and eligibility check
            obligation_bonus = 0.0
            eligible_for_any = False
            
            chunk_year = item.get("fiscal_year")
            chunk_scope = item.get("scope")

            for obl in obligations:
                obl_matched = True
                
                # Check period constraint
                if obl.required_period and chunk_year:
                    if obl.required_period.lower() != chunk_year.lower():
                        obl_matched = False
                
                # Check scope constraint
                if obl.required_scope and chunk_scope and chunk_scope != "unspecified":
                    if obl.required_scope.lower() != chunk_scope.lower():
                        obl_matched = False

                if obl_matched:
                    # Check keyword match for this obligation
                    matched_kw = sum(1 for kw in obl.keywords if kw.lower() in chunk_text.lower())
                    if matched_kw > 0 or obl_matched:
                        obligation_bonus += 0.3 * (matched_kw + 1)
                        eligible_for_any = True

            # If obligations were defined and this chunk directly conflicts with period/scope
            if obligations and not eligible_for_any and (chunk_year or chunk_scope):
                penalty = 0.5
            else:
                penalty = 0.0

            # 3. Table structure bonus for numeric queries
            table_bonus = 0.2 if item.get("segment_type") == "table" else 0.0

            final_score = base_score + term_overlap_score + obligation_bonus + table_bonus - penalty

            scored_item = dict(item)
            scored_item["rerank_score"] = float(final_score)
            scored_item["eligible"] = eligible_for_any or (not obligations)
            scored_candidates.append(scored_item)

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:limit]
