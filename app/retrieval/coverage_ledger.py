from typing import List, Dict, Any, Tuple
from app.schemas.internal_models import EvidenceObligation
from app.storage.vector_store import vector_storage

class EvidenceCoverageLedger:
    """
    Manages evidence coverage across obligations and drives targeted gap filling:
    - Verifies which obligations are supported by currently retrieved candidates
    - Identifies missing periods, missing explanations, or missing mitigation steps
    - Executes targeted, bounded queries to retrieve missing evidence
    """

    def evaluate_coverage(
        self,
        obligations: List[EvidenceObligation],
        candidates: List[Dict[str, Any]]
    ) -> List[EvidenceObligation]:
        for obl in obligations:
            supporting_ids = []
            
            for c in candidates:
                text = c.get("text", "").lower()
                c_year = c.get("fiscal_year")
                c_scope = c.get("scope")
                
                # Check period compatibility
                if obl.required_period and c_year:
                    if obl.required_period.lower() != c_year.lower():
                        continue
                        
                # Check scope compatibility
                if obl.required_scope and c_scope and c_scope != "unspecified":
                    if obl.required_scope.lower() != c_scope.lower():
                        continue

                # Check keyword or intent match
                kw_matches = sum(1 for kw in obl.keywords if kw.lower() in text)
                if kw_matches > 0 or not obl.keywords:
                    supporting_ids.append(c["chunk_id"])

            obl.supporting_chunk_ids = list(set(supporting_ids))
            if len(supporting_ids) >= 2:
                obl.coverage_status = "fully_covered"
            elif len(supporting_ids) == 1:
                obl.coverage_status = "partially_covered"
            else:
                obl.coverage_status = "uncovered"

        return obligations

    def fill_gaps(
        self,
        collection_name: str,
        obligations: List[EvidenceObligation],
        current_candidates: List[Dict[str, Any]],
        base_query: str,
        max_gap_queries: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[EvidenceObligation]]:
        uncovered = [o for o in obligations if o.coverage_status == "uncovered"]
        if not uncovered:
            return current_candidates, obligations

        existing_cids = {c["chunk_id"] for c in current_candidates}
        additional_chunks: List[Dict[str, Any]] = []

        for obl in uncovered[:max_gap_queries]:
            obl.retrieval_attempts += 1
            
            # Construct targeted sub-query
            targeted_terms = [base_query]
            if obl.required_period:
                targeted_terms.append(obl.required_period)
            if obl.keywords:
                targeted_terms.extend(obl.keywords[:3])
            targeted_query = " ".join(targeted_terms)

            filters = {}
            if obl.required_period:
                filters["fiscal_year"] = obl.required_period
            if obl.required_scope:
                filters["scope"] = obl.required_scope

            # Search with filters
            gap_results = vector_storage.hybrid_search(
                collection_name=collection_name,
                query=targeted_query,
                limit=5,
                filters=filters if filters else None
            )

            # If filtered search yielded nothing, try without hard filters
            if not gap_results and filters:
                gap_results = vector_storage.hybrid_search(
                    collection_name=collection_name,
                    query=targeted_query,
                    limit=5
                )

            for item in gap_results:
                if item["chunk_id"] not in existing_cids:
                    existing_cids.add(item["chunk_id"])
                    additional_chunks.append(item)

        # Merge and re-evaluate coverage
        updated_candidates = current_candidates + additional_chunks
        updated_obligations = self.evaluate_coverage(obligations, updated_candidates)
        
        return updated_candidates, updated_obligations
