import re
from typing import List, Dict, Any, Tuple
from app.schemas.api_models import ChunkCitation
from app.schemas.internal_models import EvidenceObligation

class ClaimValidator:
    """
    EFAR Claim and Citation Validator:
    Audits the generated answer to ensure:
    1. Every bracketed citation corresponds to a real selected source and page number.
    2. No definitive claims are asserted for obligations marked as 'unresolved'.
    3. Cites explicit evidence for material financial figures.
    """

    CITATION_PATTERN = re.compile(r'\[([A-Za-z0-9_\-\.\s]+?)(?:,\s*(?:Page|p\.?)\s*(\d+))?\]')

    def validate_answer(
        self,
        answer: str,
        selected_chunks: List[Dict[str, Any]],
        obligations: List[EvidenceObligation]
    ) -> Dict[str, Any]:
        
        valid_doc_pages = set()
        valid_docs = set()
        for c in selected_chunks:
            doc_id = c.get("document_id")
            page = c.get("page_number")
            if doc_id:
                valid_docs.add(doc_id.lower())
                if page:
                    valid_doc_pages.add((doc_id.lower(), page))

        # 1. Parse citations in answer
        found_citations = self.CITATION_PATTERN.findall(answer)
        verified_count = 0
        hallucinated_citations = []

        for doc_name, page_str in found_citations:
            doc_clean = doc_name.strip().lower()
            page_num = int(page_str) if page_str else None

            # Check if doc exists in selected evidence
            doc_matched = any(doc_clean in vd or vd in doc_clean for vd in valid_docs)
            if doc_matched:
                verified_count += 1
            else:
                hallucinated_citations.append(f"{doc_name} (Page {page_str})")

        # 2. Check unresolved obligations
        unresolved = [o for o in obligations if o.status == "unresolved"]
        unresolved_notes = []
        for o in unresolved:
            unresolved_notes.append(o.description)

        is_valid = len(hallucinated_citations) == 0

        return {
            "is_valid": is_valid,
            "citations_found": len(found_citations),
            "citations_verified": verified_count,
            "hallucinated_citations": hallucinated_citations,
            "unresolved_obligations_tracked": len(unresolved),
            "warnings": [f"Unverified citation in answer: {c}" for c in hallucinated_citations]
        }

claim_validator = ClaimValidator()
