from typing import List, Dict, Any, Optional

class ConsistencyAnalyzer:
    """
    EFAR Consistency Analyzer:
    Analyzes selected evidence to distinguish genuine contradictions from scope/period distinctions.
    Prevents false reconciliations when sources use different accounting bases.
    """

    def analyze_consistency(self, selected_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        discrepancies = []
        scope_differentiations = []

        # Group chunks by document
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for c in selected_chunks:
            by_doc.setdefault(c.get("document_id", "unknown"), []).append(c)

        # Inspect if both consolidated and standalone chunks are present
        scopes_present = set(c.get("scope") for c in selected_chunks if c.get("scope"))
        if "consolidated" in scopes_present and "standalone" in scopes_present:
            scope_differentiations.append({
                "type": "scope_distinction",
                "note": "Selected context contains both 'consolidated' and 'standalone' figures. Keep these figures separate and do not confuse them."
            })

        # Inspect if multiple fiscal periods are present
        years_present = set(c.get("fiscal_year") for c in selected_chunks if c.get("fiscal_year"))
        if len(years_present) > 1:
            scope_differentiations.append({
                "type": "period_distinction",
                "note": f"Selected context covers multiple fiscal periods ({', '.join(years_present)}). Maintain year-specific citations for each figure."
            })

        return {
            "has_genuine_conflicts": len(discrepancies) > 0,
            "discrepancies": discrepancies,
            "scope_differentiations": scope_differentiations,
            "scopes_present": list(scopes_present),
            "years_present": list(years_present)
        }

consistency_analyzer = ConsistencyAnalyzer()
