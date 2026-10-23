from typing import List, Dict, Any, Tuple
from app.schemas.internal_models import EvidenceObligation
from app.storage.vector_store import vector_storage
from app.retrieval.support_matrix import support_matrix_builder, SupportRelation

class BoundedRecoveryEngine:
    """
    EFAR Bounded Recovery Engine:
    Executes focused, budgeted secondary retrieval strictly for unresolved obligations.
    Stops when coverage is adequate or the budget is exhausted.
    """

    def recover_missing_evidence(
        self,
        collection_name: str,
        base_query: str,
        unresolved_obligations: List[EvidenceObligation],
        current_selected_chunks: List[Dict[str, Any]],
        max_recovery_rounds: int = 1,
        max_candidates_per_obl: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[EvidenceObligation], Dict[str, Any]]:
        
        if not unresolved_obligations or max_recovery_rounds <= 0:
            return current_selected_chunks, unresolved_obligations, {"rounds_run": 0, "recovered_chunks": 0}

        existing_cids = {c["chunk_id"] for c in current_selected_chunks}
        recovered_chunks: List[Dict[str, Any]] = []
        updated_obligations: List[EvidenceObligation] = []

        for obl in unresolved_obligations:
            obl_copy = obl.model_copy()
            obl_copy.retrieval_attempts += 1

            # Build targeted query
            terms = [base_query]
            if obl_copy.required_period:
                terms.append(obl_copy.required_period)
            if obl_copy.required_entity:
                terms.append(obl_copy.required_entity)
            if obl_copy.keywords:
                terms.extend(obl_copy.keywords[:3])

            targeted_query = " ".join(terms)
            filters = {}
            if obl_copy.required_period:
                filters["fiscal_year"] = obl_copy.required_period
            if obl_copy.required_scope:
                filters["scope"] = obl_copy.required_scope

            # 1. Targeted hybrid search
            new_candidates = vector_storage.hybrid_search(
                collection_name=collection_name,
                query=targeted_query,
                limit=max_candidates_per_obl,
                filters=filters if filters else None
            )

            # Fallback without hard filters if empty
            if not new_candidates and filters:
                new_candidates = vector_storage.hybrid_search(
                    collection_name=collection_name,
                    query=targeted_query,
                    limit=max_candidates_per_obl
                )

            # 2. Assess newly discovered candidates
            for cand in new_candidates:
                if cand["chunk_id"] in existing_cids:
                    continue

                assessment = support_matrix_builder._assess_pair(cand, obl_copy)
                if assessment.relation in [SupportRelation.DIRECT_SUPPORT, SupportRelation.PARTIAL_SUPPORT]:
                    existing_cids.add(cand["chunk_id"])
                    recovered_chunks.append(cand)
                    obl_copy.supporting_chunk_ids.append(cand["chunk_id"])
                    obl_copy.status = "supported"
                    obl_copy.coverage_status = "partially_covered"
                    break # satisfied for this round

            updated_obligations.append(obl_copy)

        combined_selected = current_selected_chunks + recovered_chunks

        report = {
            "rounds_run": 1,
            "recovered_chunks": len(recovered_chunks),
            "recovered_chunk_ids": [c["chunk_id"] for c in recovered_chunks],
            "obligations_resolved_in_recovery": sum(1 for o in updated_obligations if o.status == "supported")
        }

        return combined_selected, updated_obligations, report

recovery_engine = BoundedRecoveryEngine()
