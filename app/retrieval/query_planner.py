import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
from app.schemas.internal_models import EvidenceObligation, QueryPlan

class QueryPlanner:
    """
    EFAR Query Compiler:
    Decomposes incoming queries into testable Evidence Obligations with explicit
    hard constraints, soft preferences, and adaptive query plans.
    """
    
    YEAR_PATTERN = re.compile(r'\b(FY\s*20\d{2}|20\d{2}(?:-\d{2})?)\b', re.IGNORECASE)
    ENTITY_PATTERN = re.compile(r'\b([A-Z][a-zA-Z0-9&.\s]{2,30}\s+(?:Limited|Ltd|Inc|Corp|Corporation|Bank|Group|Holdings))\b', re.IGNORECASE)

    # Sub-topic keywords for financial distress / deep analysis queries
    SUB_TOPIC_KEYWORDS = {
        "lender_actions": ["lender", "npa", "recall", "sarfaesi", "arc", "drt", "nclt", "pledge", "possession", "recovery suit"],
        "payment_defaults": ["default", "principal default", "interest default", "ncd", "unpaid", "statutory dues"],
        "auditor_concerns": ["auditor", "qualification", "qualified", "going concern", "material uncertainty", "internal control", "penal interest"],
        "market_governance": ["delist", "delisting", "suspended", "disqualification", "section 164", "company secretary", "compliance"],
        "financial_performance": ["turnover", "loss", "profit", "net worth", "accumulated loss", "finance cost", "borrowing"]
    }

    def compile_plan(self, query: str, query_id: Optional[str] = None) -> QueryPlan:
        q_id = query_id or f"qp_{uuid.uuid4().hex[:8]}"
        q_lower = query.lower()

        # 1. Detect entities mentioned
        entities_found = []
        entity_match = self.ENTITY_PATTERN.search(query)
        if entity_match:
            candidate = entity_match.group(1).strip()
            if candidate and candidate not in entities_found:
                entities_found.append(candidate)
        else:
            # Check common benchmark entities
            if "gammon" in q_lower:
                entities_found.append("Gammon India Limited")
            elif "aldar" in q_lower:
                entities_found.append("Aldar")

        # 2. Detect years/periods
        years_found = []
        for m in self.YEAR_PATTERN.finditer(query):
            val = m.group(1).upper().replace(" ", "")
            formatted = val if val.startswith("FY") else f"FY{val}"
            if formatted not in years_found:
                years_found.append(formatted)

        # 3. Detect scope constraints
        scope = None
        if "consolidated" in q_lower or "group" in q_lower:
            scope = "consolidated"
        elif "standalone" in q_lower:
            scope = "standalone"
        elif "subsidiary" in q_lower or "branch" in q_lower:
            scope = "subsidiary"

        # 4. Detect intent & multi-topic flags
        is_comparison = any(w in q_lower for w in ["compare", "comparison", "versus", "vs", "difference", "change", "increased", "decreased"])
        needs_explanation = any(w in q_lower for w in ["explain", "explanation", "why", "cause", "reason", "driver", "factors"])
        needs_mitigation = any(w in q_lower for w in ["mitigat", "action", "measure", "control", "manage", "response", "settlement", "ots"])
        is_numeric = any(w in q_lower for w in ["how much", "amount", "figure", "value", "exposure", "revenue", "loss", "ratio", "percentage", "margin"])

        # Check multi-topic distress inquiry (e.g. Question A01)
        sub_topics_requested = [
            topic for topic, kws in self.SUB_TOPIC_KEYWORDS.items()
            if any(kw in q_lower for kw in kws)
        ]

        obligations: List[EvidenceObligation] = []
        hard_constraints = {}
        if entities_found:
            hard_constraints["entity"] = entities_found[0]
        if len(years_found) == 1:
            hard_constraints["fiscal_year"] = years_found[0]
        if scope:
            hard_constraints["scope"] = scope

        # Scenario 1: Multi-topic deep investigation (e.g. Gammon India red flags)
        if len(sub_topics_requested) >= 2:
            intent = "deep_investigation"
            for topic in sub_topics_requested:
                topic_desc = topic.replace("_", " ").title()
                obl = EvidenceObligation(
                    obligation_id=f"obl_{topic}",
                    description=f"Specific evidence and metrics regarding {topic_desc}",
                    requirement_type="fact",
                    required_entity=entities_found[0] if entities_found else None,
                    required_period=years_found[0] if years_found else None,
                    required_scope=scope,
                    required_concepts=[topic],
                    hard_constraints=dict(hard_constraints),
                    keywords=self.SUB_TOPIC_KEYWORDS[topic][:5],
                    weight=1.2
                )
                obligations.append(obl)

            # Overall bottom-line / severity assessment
            obligations.append(EvidenceObligation(
                obligation_id="obl_overall_assessment",
                description="Evaluation of overall severity, going concern uncertainty, and recovery prospects",
                requirement_type="qualification",
                required_entity=entities_found[0] if entities_found else None,
                required_period=years_found[0] if years_found else None,
                required_concepts=["going_concern", "severity"],
                keywords=["going concern", "uncertainty", "severity", "settlement"],
                weight=1.0
            ))

        # Scenario 2: Cross-period comparison
        elif len(years_found) >= 2 or (is_comparison and len(years_found) == 1):
            intent = "comparison"
            for yr in years_found:
                hc = dict(hard_constraints)
                hc["fiscal_year"] = yr
                obligations.append(EvidenceObligation(
                    obligation_id=f"obl_period_{yr}",
                    description=f"Evidence for {yr} specific metrics, figures, and definitions",
                    requirement_type="numeric_fact" if is_numeric else "fact",
                    required_entity=entities_found[0] if entities_found else None,
                    required_period=yr,
                    required_scope=scope,
                    hard_constraints=hc,
                    keywords=[yr] + [w for w in ["exposure", "credit", "risk", "revenue"] if w in q_lower],
                    weight=1.5
                ))

            if is_comparison:
                obligations.append(EvidenceObligation(
                    obligation_id="obl_delta_comparison",
                    description=f"Direct comparison or calculation of delta between {' and '.join(years_found)}",
                    requirement_type="comparison",
                    required_scope=scope,
                    keywords=["change", "difference", "increase", "decrease"] + years_found,
                    weight=1.0
                ))

        # Scenario 3: Single Period / Focused Query
        elif len(years_found) == 1:
            intent = "factual"
            yr = years_found[0]
            hc = dict(hard_constraints)
            hc["fiscal_year"] = yr
            obligations.append(EvidenceObligation(
                obligation_id=f"obl_period_{yr}",
                description=f"Evidence for {yr} reporting data and facts",
                requirement_type="numeric_fact" if is_numeric else "fact",
                required_entity=entities_found[0] if entities_found else None,
                required_period=yr,
                required_scope=scope,
                hard_constraints=hc,
                keywords=[yr],
                weight=1.0
            ))
        else:
            intent = "informational"
            obligations.append(EvidenceObligation(
                obligation_id="obl_core_fact",
                description="Core factual evidence supporting the main query assertion",
                requirement_type="fact",
                required_entity=entities_found[0] if entities_found else None,
                required_scope=scope,
                hard_constraints=hard_constraints,
                keywords=q_lower.split()[:5],
                weight=1.0
            ))

        # Supplemental obligations (Explanation / Mitigation)
        if needs_explanation and not any(o.obligation_id == "obl_explanation" for o in obligations):
            obligations.append(EvidenceObligation(
                obligation_id="obl_explanation",
                description="Reported explanations, causes, or commentary justifying changes/metrics",
                requirement_type="explanation",
                required_entity=entities_found[0] if entities_found else None,
                required_scope=scope,
                keywords=["reason", "driver", "due to", "because", "increase", "decrease", "caused by"],
                weight=1.0
            ))

        if needs_mitigation and not any(o.obligation_id == "obl_mitigation" for o in obligations):
            obligations.append(EvidenceObligation(
                obligation_id="obl_mitigation",
                description="Risk mitigation measures, management actions, and controls reported",
                requirement_type="mitigation",
                required_entity=entities_found[0] if entities_found else None,
                required_scope=scope,
                keywords=["mitigation", "management", "risk control", "policy", "hedge", "strategy"],
                weight=1.0
            ))

        return QueryPlan(
            query_id=q_id,
            original_query=query,
            intent=intent,
            entity_constraints=entities_found,
            scope_constraints=scope,
            period_constraints=years_found,
            obligations=obligations,
            retrieval_budget={
                "candidate_depth": 30 if intent == "deep_investigation" else 20,
                "max_context_chunks": 12 if intent == "deep_investigation" else 8,
                "max_recovery_rounds": 1
            }
        )

    def plan_query(self, query: str) -> Tuple[Dict[str, Any], List[EvidenceObligation]]:
        """
        Backwards-compatible wrapper returning (query_meta, obligations).
        """
        plan = self.compile_plan(query)
        query_meta = {
            "query_id": plan.query_id,
            "intent": plan.intent,
            "years_detected": plan.period_constraints,
            "entities_detected": plan.entity_constraints,
            "scope_detected": plan.scope_constraints,
            "is_comparison": plan.intent == "comparison",
            "needs_explanation": any(o.requirement_type == "explanation" for o in plan.obligations),
            "needs_mitigation": any(o.requirement_type == "mitigation" for o in plan.obligations),
            "total_obligations": len(plan.obligations),
            "query_plan": plan
        }
        return query_meta, plan.obligations

query_planner = QueryPlanner()
