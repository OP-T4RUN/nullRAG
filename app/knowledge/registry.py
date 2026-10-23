import re
from typing import Dict, List, Optional, Tuple

class ConceptRegistry:
    """
    Lightweight concept registry for open-world attributes.
    Maps observed raw attribute names and synonyms to canonical business/financial concepts.
    Preserves unfamiliar attributes as open-world concepts without requiring DB schema changes.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConceptRegistry, cls).__new__(cls)
            cls._instance._init_registry()
        return cls._instance

    def _init_registry(self):
        # Canonical concept -> list of regex patterns / synonyms
        self.mappings: Dict[str, List[re.Pattern]] = {
            "credit_risk_exposure": [
                re.compile(r'\b(credit\s+risk\s+exposure|credit\s+exposure|cva\s+exposure|counterparty\s+exposure)\b', re.IGNORECASE)
            ],
            "npa_status": [
                re.compile(r'\b(npa|non-performing\s+asset|recalled\s+facility|recalled\s+debt|loan\s+default)\b', re.IGNORECASE)
            ],
            "net_loss_or_profit": [
                re.compile(r'\b(net\s+loss|profit\s+after\s+tax|loss\s+after\s+tax|net\s+profit|pat)\b', re.IGNORECASE)
            ],
            "turnover_revenue": [
                re.compile(r'\b(turnover|revenue|total\s+income|sales)\b', re.IGNORECASE)
            ],
            "finance_cost": [
                re.compile(r'\b(finance\s+cost|interest\s+cost|interest\s+expense)\b', re.IGNORECASE)
            ],
            "auditor_qualification": [
                re.compile(r'\b(auditor\s+qualification|qualified\s+opinion|going\s+concern|material\s+uncertainty)\b', re.IGNORECASE)
            ],
            "restructuring_ots": [
                re.compile(r'\b(one\s+time\s+settlement|ots|cdr|sdr|resolution\s+plan|ica)\b', re.IGNORECASE)
            ],
            "governance_delisting": [
                re.compile(r'\b(delist|delisting|suspended|sat\s+order|director\s+disqualification|section\s+164)\b', re.IGNORECASE)
            ],
            "capital_adequacy": [
                re.compile(r'\b(capital\s+adequacy|car|tier\s+1\s+capital|cet1|crar)\b', re.IGNORECASE)
            ],
            "risk_mitigation": [
                re.compile(r'\b(mitigation|risk\s+control|hedging|collateral|stress\s+testing)\b', re.IGNORECASE)
            ]
        }
        self.dynamic_concepts: Dict[str, str] = {} # raw_name -> candidate_canonical

    def canonicalize(self, raw_attribute_name: str) -> Tuple[str, float]:
        """
        Attempts to map raw attribute name to canonical concept.
        Returns (canonical_name, confidence).
        If unknown, returns normalized raw name with confidence 0.7 (open-world principle).
        """
        cleaned = raw_attribute_name.strip().lower()
        
        # Check predefined canonical concepts
        for canonical, patterns in self.mappings.items():
            for p in patterns:
                if p.search(cleaned):
                    return canonical, 0.95

        # Check registered dynamic concepts
        if cleaned in self.dynamic_concepts:
            return self.dynamic_concepts[cleaned], 0.85

        # Fallback to normalized slug
        slug = re.sub(r'[^a-z0-9_]+', '_', cleaned).strip('_')
        return slug or "unclassified_attribute", 0.70

    def register_concept(self, raw_attribute_name: str, canonical_concept: str):
        self.dynamic_concepts[raw_attribute_name.strip().lower()] = canonical_concept

concept_registry = ConceptRegistry()
