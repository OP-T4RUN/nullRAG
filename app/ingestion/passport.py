import re
import hashlib
from typing import List, Optional, Dict, Any
from datasketch import MinHash

from app.schemas.internal_models import ParsedSegment, DocumentPassport

class PassportExtractor:
    """
    Extracts high-fidelity Document Passport metadata:
    - Entity / Company Name
    - Fiscal Year / Reporting Period
    - Document Type (10-K, Risk Disclosure, Annual Report, etc.)
    - Scope (Consolidated, Subsidiary, Standalone)
    - Currency and Units
    - MinHash and Content Hash for deduplication
    """

    YEAR_PATTERN = re.compile(r'\b(FY\s*20\d{2}|20\d{2})\b', re.IGNORECASE)
    CONSOLIDATED_PATTERN = re.compile(r'\b(consolidated|group|consolidated financial statements)\b', re.IGNORECASE)
    SUBSIDIARY_PATTERN = re.compile(r'\b(subsidiary|division|standalone|branch)\b', re.IGNORECASE)
    
    DOC_TYPES = {
        "annual_report": re.compile(r'\b(annual report|form 10-k|10-k)\b', re.IGNORECASE),
        "quarterly_report": re.compile(r'\b(quarterly report|form 10-q|10-q|q[1-4]\s*20\d{2})\b', re.IGNORECASE),
        "risk_disclosure": re.compile(r'\b(risk disclosure|risk report|pillar 3|credit risk)\b', re.IGNORECASE),
        "financial_statement": re.compile(r'\b(financial statements|balance sheet|income statement)\b', re.IGNORECASE),
        "investor_presentation": re.compile(r'\b(investor presentation|earnings presentation)\b', re.IGNORECASE),
    }

    CURRENCY_PATTERN = re.compile(r'\b(USD|\$|EUR|€|GBP|£|INR|₹|JPY|¥)\b')
    UNITS_PATTERN = re.compile(r'\b(in millions|in billions|in thousands|millions|billions|thousands)\b', re.IGNORECASE)

    ENTITY_PATTERNS = [
        re.compile(r'([A-Z][A-Za-z0-9&.,\s]{2,40}\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|PLC|Bank|Group|Technologies|Holdings))', re.MULTILINE),
        re.compile(r'^([A-Z][A-Za-z0-9&.,\s]{2,30})\s*(?:\n|$)', re.MULTILINE)
    ]

    def extract_passport(self, collection_name: str, document_id: str, filename: str, segments: List[ParsedSegment]) -> DocumentPassport:
        # Concentrate primarily on early pages (page 1-3) for document title and identity
        front_segments = [s for s in segments if s.page_number <= 3]
        front_text = " ".join([s.text for s in front_segments])
        all_text = " ".join([s.text for s in segments])

        # 1. Content Hash & MinHash
        content_hash = hashlib.sha256(all_text.encode("utf-8")).hexdigest()
        minhash = self._compute_minhash(all_text)

        # 2. Fiscal Year / Period
        fiscal_year = self._extract_fiscal_year(front_text, filename, all_text)

        # 3. Scope (Consolidated vs Subsidiary)
        scope = self._extract_scope(front_text, all_text)

        # 4. Document Type
        doc_type = self._extract_document_type(front_text, filename)

        # 5. Entity Name
        entity = self._extract_entity(front_text, filename)

        # 6. Currency and Units
        currency = self._extract_currency(front_text, all_text)
        units = self._extract_units(front_text, all_text)

        page_count = max([s.page_number for s in segments], default=1)

        # Confidence calculation
        confidence = 0.5
        if fiscal_year: confidence += 0.2
        if entity: confidence += 0.15
        if scope: confidence += 0.15

        return DocumentPassport(
            document_id=document_id,
            collection_name=collection_name,
            filename=filename,
            entity=entity,
            fiscal_year=fiscal_year,
            publication_date=fiscal_year,
            document_type=doc_type,
            scope=scope,
            version=None,
            currency=currency,
            units=units,
            content_hash=content_hash,
            page_count=page_count,
            metadata_confidence=min(1.0, confidence),
            extracted_metadata={
                "minhash_digest": minhash.digest().tolist(),
                "front_title_preview": front_text[:200]
            }
        )

    def _compute_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        m = MinHash(num_perm=num_perm)
        # 3-word shingles
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3])
            m.update(shingle.encode("utf-8"))
        return m

    def _extract_fiscal_year(self, front_text: str, filename: str, all_text: str) -> Optional[str]:
        # Check filename first
        match_fn = self.YEAR_PATTERN.search(filename)
        if match_fn:
            yr = match_fn.group(1).upper().replace(" ", "")
            return yr if yr.startswith("FY") else f"FY{yr}"

        # Check front pages
        match_front = self.YEAR_PATTERN.search(front_text)
        if match_front:
            yr = match_front.group(1).upper().replace(" ", "")
            return yr if yr.startswith("FY") else f"FY{yr}"

        # Check all text
        match_all = self.YEAR_PATTERN.search(all_text[:5000])
        if match_all:
            yr = match_all.group(1).upper().replace(" ", "")
            return yr if yr.startswith("FY") else f"FY{yr}"

        return None

    def _extract_scope(self, front_text: str, all_text: str) -> str:
        if self.CONSOLIDATED_PATTERN.search(front_text) or self.CONSOLIDATED_PATTERN.search(all_text[:3000]):
            return "consolidated"
        if self.SUBSIDIARY_PATTERN.search(front_text) or self.SUBSIDIARY_PATTERN.search(all_text[:3000]):
            return "subsidiary"
        return "unspecified"

    def _extract_document_type(self, front_text: str, filename: str) -> str:
        combined = f"{filename}\n{front_text}".lower()
        for dtype, pattern in self.DOC_TYPES.items():
            if pattern.search(combined):
                return dtype
        return "general_report"

    def _extract_entity(self, front_text: str, filename: str) -> Optional[str]:
        for pattern in self.ENTITY_PATTERNS:
            match = pattern.search(front_text)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 2 and not candidate.lower().startswith("table") and not candidate.lower().startswith("page"):
                    return candidate
        # Fallback to cleaned filename
        clean_fn = re.sub(r'[_.-]', ' ', filename).replace("pdf", "").strip()
        words = clean_fn.split()
        if words:
            return words[0]
        return None

    def _extract_currency(self, front_text: str, all_text: str) -> Optional[str]:
        match = self.CURRENCY_PATTERN.search(front_text) or self.CURRENCY_PATTERN.search(all_text[:3000])
        return match.group(1) if match else "USD"

    def _extract_units(self, front_text: str, all_text: str) -> Optional[str]:
        match = self.UNITS_PATTERN.search(front_text) or self.UNITS_PATTERN.search(all_text[:3000])
        return match.group(1).lower() if match else None
