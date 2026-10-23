import re
from typing import List, Dict, Any, Optional

class NumericVerifier:
    """
    Verifies numerical claims, currency units, and cross-period comparisons.
    Provides deterministic math verification to prevent LLM arithmetic hallucinations.
    """
    
    NUMBER_PATTERN = re.compile(r'([$€£₹]?)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(million|billion|thousand|%)?', re.IGNORECASE)

    def extract_metrics(self, text: str) -> List[Dict[str, Any]]:
        metrics = []
        for m in self.NUMBER_PATTERN.finditer(text):
            currency = m.group(1)
            raw_val = m.group(2).replace(",", "")
            unit = m.group(3) or ""
            try:
                num = float(raw_val)
                metrics.append({
                    "raw_string": m.group(0),
                    "value": num,
                    "currency": currency,
                    "unit": unit.lower(),
                })
            except ValueError:
                continue
        return metrics

    def verify_comparison(
        self,
        val_a: float,
        unit_a: str,
        val_b: float,
        unit_b: str,
        label_a: str = "Period A",
        label_b: str = "Period B"
    ) -> Dict[str, Any]:
        """
        Deterministic arithmetic comparison
        """
        if unit_a.lower() != unit_b.lower():
            return {
                "compatible": False,
                "warning": f"Unit mismatch: {label_a} is in '{unit_a}' while {label_b} is in '{unit_b}'."
            }
            
        diff = val_b - val_a
        pct_change = ((diff / val_a) * 100) if val_a != 0 else 0.0
        
        return {
            "compatible": True,
            "val_a": val_a,
            "val_b": val_b,
            "unit": unit_a,
            "absolute_change": diff,
            "percentage_change": round(pct_change, 2),
            "summary": f"Change from {label_a} ({val_a} {unit_a}) to {label_b} ({val_b} {unit_b}) is {diff:+.2f} {unit_a} ({pct_change:+.2f}%)."
        }
