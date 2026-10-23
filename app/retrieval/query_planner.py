import re
from typing import List, Dict, Any, Tuple
from app.schemas.internal_models import EvidenceObligation

class QueryPlanner:
    """
    Decomposes incoming query into testable Evidence Obligations.
    Prevents one-shot shallow retrieval failure by explicitly identifying:
    - Distinct reporting periods (e.g. FY2024 vs FY2025)
    - Distinct scopes (Consolidated vs Subsidiary)
    - Comparison dimensions, explanations, and mitigation requirements
    """
    
    YEAR_PATTERN = re.compile(r'\b(FY\s*20\d{2}|20\d{2})\b', re.IGNORECASE)
    
    def plan_query(self, query: str) -> Tuple[Dict[str, Any], List[EvidenceObligation]]:
        q_lower = query.lower()
        
        # 1. Detect all years/periods mentioned
        years_found = []
        for m in self.YEAR_PATTERN.finditer(query):
            val = m.group(1).upper().replace(" ", "")
            formatted = val if val.startswith("FY") else f"FY{val}"
            if formatted not in years_found:
                years_found.append(formatted)
                
        # 2. Detect scope constraints
        scope = None
        if "consolidated" in q_lower or "group" in q_lower:
            scope = "consolidated"
        elif "subsidiary" in q_lower or "branch" in q_lower:
            scope = "subsidiary"
            
        # 3. Detect intent categories
        is_comparison = any(w in q_lower for w in ["compare", "comparison", "versus", "vs", "difference", "change", "increased", "decreased"])
        needs_explanation = any(w in q_lower for w in ["explain", "explanation", "why", "cause", "reason", "driver", "factors"])
        needs_mitigation = any(w in q_lower for w in ["mitigat", "action", "measure", "control", "manage", "response"])
        is_numeric = any(w in q_lower for w in ["how much", "amount", "figure", "value", "exposure", "revenue", "loss", "ratio", "percentage", "margin"])

        obligations: List[EvidenceObligation] = []
        
        # Scenario A: Comparison across periods (e.g. FY2024 vs FY2025)
        if len(years_found) >= 2 or (is_comparison and len(years_found) == 1):
            for i, yr in enumerate(years_found):
                obligations.append(EvidenceObligation(
                    obligation_id=f"obl_period_{yr}",
                    description=f"Evidence for {yr} specific metrics, figures, and reporting definitions",
                    required_evidence_type="numeric_fact" if is_numeric else "period_fact",
                    required_period=yr,
                    required_scope=scope,
                    keywords=[yr] + [w for w in ["exposure", "credit", "risk", "revenue"] if w in q_lower]
                ))
            
            if is_comparison:
                obligations.append(EvidenceObligation(
                    obligation_id="obl_delta_comparison",
                    description=f"Direct comparison or calculation of delta between {' and '.join(years_found)}",
                    required_evidence_type="comparison",
                    required_scope=scope,
                    keywords=["change", "difference", "increase", "decrease"] + years_found
                ))
                
        elif len(years_found) == 1:
            yr = years_found[0]
            obligations.append(EvidenceObligation(
                obligation_id=f"obl_period_{yr}",
                description=f"Evidence for {yr} reporting data and facts",
                required_evidence_type="numeric_fact" if is_numeric else "general_fact",
                required_period=yr,
                required_scope=scope,
                keywords=[yr]
            ))
        else:
            # General fact obligation
            obligations.append(EvidenceObligation(
                obligation_id="obl_core_fact",
                description="Core factual evidence supporting the main query assertion",
                required_evidence_type="general_fact",
                required_scope=scope,
                keywords=q_lower.split()[:5]
            ))

        if needs_explanation:
            obligations.append(EvidenceObligation(
                obligation_id="obl_explanation",
                description="Reported explanations, causes, or commentary justifying changes/metrics",
                required_evidence_type="explanation",
                required_scope=scope,
                keywords=["reason", "driver", "due to", "because", "increase", "decrease", "caused by"]
            ))

        if needs_mitigation:
            obligations.append(EvidenceObligation(
                obligation_id="obl_mitigation",
                description="Risk mitigation measures, management actions, and controls reported",
                required_evidence_type="mitigation",
                required_scope=scope,
                keywords=["mitigation", "management", "risk control", "policy", "hedge", "strategy"]
            ))

        query_meta = {
            "years_detected": years_found,
            "scope_detected": scope,
            "is_comparison": is_comparison,
            "needs_explanation": needs_explanation,
            "needs_mitigation": needs_mitigation,
            "total_obligations": len(obligations)
        }

        return query_meta, obligations
