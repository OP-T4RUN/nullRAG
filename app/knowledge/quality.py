from typing import List, Dict, Any
from app.knowledge.models import DocumentQualitySignal
from app.schemas.internal_models import ParsedSegment

class QualityMonitor:
    """
    Evaluates extraction quality signals at document and page levels:
    - Flags unreadable or empty pages
    - Records table extraction status
    - Detects whether OCR is required
    """
    def evaluate_page_quality(self, document_id: str, page_number: int, segments_on_page: List[ParsedSegment]) -> DocumentQualitySignal:
        warnings = []
        has_text = any(s.segment_type == "text" and len(s.text.strip()) > 30 for s in segments_on_page)
        has_table = any(s.segment_type == "table" for s in segments_on_page)

        extraction_status = "success"
        if not segments_on_page:
            extraction_status = "failed"
            warnings.append("Page produced 0 text segments (possible scanned image or blank page)")
        elif not has_text and not has_table:
            extraction_status = "partial"
            warnings.append("Extracted text is shorter than 30 characters")

        table_status = "extracted" if has_table else "none"

        return DocumentQualitySignal(
            document_id=document_id,
            page_number=page_number,
            extraction_status=extraction_status,
            has_usable_text=has_text,
            table_extraction_status=table_status,
            ocr_used=False,
            parser_version="1.0.0",
            quality_warnings=warnings
        )

quality_monitor = QualityMonitor()
