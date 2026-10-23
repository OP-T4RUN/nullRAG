from typing import List, Dict, Any, Tuple, Set
from app.schemas.internal_models import EvidenceObligation
from app.retrieval.support_matrix import SupportRelation, SupportAssessment

class CoverageConstrainedSelector:
    """
    EFAR Coverage-Constrained Evidence Selector:
    Greedily selects a complementary, non-redundant set of candidate chunks
    that collectively satisfy the active evidence obligations while controlling context bloat.
    """

    def select_evidence(
        self,
        candidates: List[Dict[str, Any]],
        obligations: List[EvidenceObligation],
        support_matrix: Dict[str, Dict[str, SupportAssessment]],
        max_context_chunks: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[EvidenceObligation], Dict[str, Any]]:
        
        cand_map = {c["chunk_id"]: c for c in candidates}
        selected_chunk_ids: List[str] = []
        selected_pages: Set[Tuple[str, int]] = set()

        # Track accumulated support per obligation: obl_id -> float
        obligation_coverage: Dict[str, float] = {o.obligation_id: 0.0 for o in obligations}
        supporting_chunks_map: Dict[str, List[str]] = {o.obligation_id: [] for o in obligations}

        # Greedy selection loop
        while len(selected_chunk_ids) < max_context_chunks:
            best_chunk_id = None
            best_marginal_gain = 0.0

            for cid, cand in cand_map.items():
                if cid in selected_chunk_ids:
                    continue

                marginal_gain = 0.0
                supports_uncovered = False

                for obl in obligations:
                    assessment = support_matrix.get(obl.obligation_id, {}).get(cid)
                    if not assessment or assessment.relation == SupportRelation.IRRELEVANT:
                        continue

                    current_cov = obligation_coverage[obl.obligation_id]
                    # Target coverage per obligation is 1.2
                    if current_cov < 1.2:
                        needed = 1.2 - current_cov
                        gain = min(needed, assessment.support_strength) * obl.weight
                        marginal_gain += gain
                        supports_uncovered = True

                if not supports_uncovered:
                    continue

                # Redundancy penalty if chunk is on the exact same page as an already selected chunk
                page_key = (cand.get("document_id"), cand.get("page_number"))
                if page_key in selected_pages:
                    marginal_gain *= 0.70  # 30% redundancy penalty

                # Table bonus for financial facts
                if cand.get("segment_type") == "table":
                    marginal_gain *= 1.15

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_chunk_id = cid

            # Stop if no candidate offers meaningful marginal gain
            if not best_chunk_id or best_marginal_gain < 0.10:
                break

            # Add best candidate
            selected_chunk_ids.append(best_chunk_id)
            best_cand = cand_map[best_chunk_id]
            selected_pages.add((best_cand.get("document_id"), best_cand.get("page_number")))

            # Update coverage
            for obl in obligations:
                assessment = support_matrix.get(obl.obligation_id, {}).get(best_chunk_id)
                if assessment and assessment.relation != SupportRelation.IRRELEVANT:
                    obligation_coverage[obl.obligation_id] += assessment.support_strength
                    supporting_chunks_map[obl.obligation_id].append(best_chunk_id)

            # Check if all obligations are satisfied (>= 1.0 support each)
            all_satisfied = all(cov >= 1.0 for cov in obligation_coverage.values())
            if all_satisfied and len(selected_chunk_ids) >= len(obligations):
                break

        # Finalize obligation statuses
        updated_obligations: List[EvidenceObligation] = []
        for obl in obligations:
            cov = obligation_coverage.get(obl.obligation_id, 0.0)
            sup_chunks = supporting_chunks_map.get(obl.obligation_id, [])
            
            obl_copy = obl.model_copy()
            obl_copy.supporting_chunk_ids = list(set(sup_chunks))
            
            if cov >= 0.85 or len(sup_chunks) >= 2:
                obl_copy.coverage_status = "fully_covered"
                obl_copy.status = "supported"
            elif cov >= 0.35 or len(sup_chunks) == 1:
                obl_copy.coverage_status = "partially_covered"
                obl_copy.status = "supported"
            else:
                obl_copy.coverage_status = "uncovered"
                obl_copy.status = "unresolved"

            updated_obligations.append(obl_copy)

        selected_chunks = [cand_map[cid] for cid in selected_chunk_ids]

        summary = {
            "total_candidates": len(candidates),
            "selected_count": len(selected_chunks),
            "obligations_total": len(obligations),
            "obligations_supported": sum(1 for o in updated_obligations if o.status == "supported"),
            "obligations_unresolved": sum(1 for o in updated_obligations if o.status == "unresolved"),
            "coverage_scores": {k: round(v, 2) for k, v in obligation_coverage.items()}
        }

        return selected_chunks, updated_obligations, summary

coverage_selector = CoverageConstrainedSelector()
