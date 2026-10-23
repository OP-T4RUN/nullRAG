import hashlib
from pathlib import Path
from typing import List, Optional
import pypdf
import pdfplumber

from app.schemas.internal_models import ParsedSegment

class DocumentParser:
    """
    Extracts text and structured tables from documents (PDF, TXT, MD)
    while preserving page numbers, structure, and provenance.
    """

    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def parse_file(self, file_path: Path, document_id: str) -> List[ParsedSegment]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(file_path, document_id)
        elif suffix in [".txt", ".md", ".json"]:
            return self._parse_text_file(file_path, document_id)
        else:
            # Fallback to text parsing
            try:
                return self._parse_text_file(file_path, document_id)
            except Exception:
                raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_pdf(self, file_path: Path, document_id: str) -> List[ParsedSegment]:
        segments: List[ParsedSegment] = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    
                    # 1. Extract structured tables first
                    tables = page.extract_tables()
                    table_segments_count = 0
                    
                    if tables:
                        for table_idx, raw_table in enumerate(tables):
                            if not raw_table or len(raw_table) < 2:
                                continue
                            
                            headers = [str(cell).strip() if cell is not None else "" for cell in raw_table[0]]
                            rows = []
                            table_lines = [f"| {' | '.join(headers)} |"]
                            table_lines.append(f"| {' | '.join(['---'] * len(headers))} |")
                            
                            for row in raw_table[1:]:
                                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                rows.append(cleaned_row)
                                table_lines.append(f"| {' | '.join(cleaned_row)} |")
                            
                            table_md = "\n".join(table_lines)
                            seg_id = f"{document_id}_p{page_num}_tbl{table_idx+1}"
                            
                            segments.append(ParsedSegment(
                                segment_id=seg_id,
                                document_id=document_id,
                                page_number=page_num,
                                segment_type="table",
                                text=table_md,
                                table_id=f"table_{page_num}_{table_idx+1}",
                                table_headers=headers,
                                table_rows=rows,
                                source_location={"page": page_num, "table_index": table_idx},
                                content_hash=self.compute_hash(table_md)
                            ))
                            table_segments_count += 1
                    
                    # 2. Extract standard page text
                    text = page.extract_text(layout=False) or ""
                    text = text.strip()
                    
                    if text:
                        seg_id = f"{document_id}_p{page_num}_text"
                        segments.append(ParsedSegment(
                            segment_id=seg_id,
                            document_id=document_id,
                            page_number=page_num,
                            segment_type="text",
                            text=text,
                            source_location={"page": page_num},
                            content_hash=self.compute_hash(text)
                        ))
                        
        except Exception as e:
            # Fallback to pypdf if pdfplumber encounters an issue
            segments = self._parse_pdf_pypdf_fallback(file_path, document_id)
            
        return segments

    def _parse_pdf_pypdf_fallback(self, file_path: Path, document_id: str) -> List[ParsedSegment]:
        segments: List[ParsedSegment] = []
        reader = pypdf.PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            text = (page.extract_text() or "").strip()
            if text:
                seg_id = f"{document_id}_p{page_num}_text"
                segments.append(ParsedSegment(
                    segment_id=seg_id,
                    document_id=document_id,
                    page_number=page_num,
                    segment_type="text",
                    text=text,
                    source_location={"page": page_num},
                    content_hash=self.compute_hash(text)
                ))
        return segments

    def _parse_text_file(self, file_path: Path, document_id: str) -> List[ParsedSegment]:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        return [
            ParsedSegment(
                segment_id=f"{document_id}_p1_text",
                document_id=document_id,
                page_number=1,
                segment_type="text",
                text=text,
                source_location={"page": 1},
                content_hash=self.compute_hash(text)
            )
        ]
