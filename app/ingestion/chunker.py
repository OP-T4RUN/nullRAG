from typing import List
from app.config import settings
from app.schemas.internal_models import ParsedSegment, DocumentPassport, IndexedChunk

class StructureAwareChunker:
    """
    Chunks document segments into retrieval units while:
    - Retaining page-number and document provenance
    - Preserving table headers with every table row/chunk
    - Attaching Passport metadata (entity, fiscal_year, scope) to chunk payloads
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, passport: DocumentPassport, segments: List[ParsedSegment]) -> List[IndexedChunk]:
        chunks: List[IndexedChunk] = []
        chunk_idx = 1

        for seg in segments:
            if seg.segment_type == "table":
                table_chunks = self._chunk_table(passport, seg, chunk_idx)
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)
            else:
                text_chunks = self._chunk_text(passport, seg, chunk_idx)
                chunks.extend(text_chunks)
                chunk_idx += len(text_chunks)

        return chunks

    def _chunk_text(self, passport: DocumentPassport, segment: ParsedSegment, start_idx: int) -> List[IndexedChunk]:
        words = segment.text.split()
        if not words:
            return []

        chunks: List[IndexedChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{passport.document_id}_p{segment.page_number}_c{start_idx + len(chunks):03d}"
            
            chunks.append(IndexedChunk(
                chunk_id=chunk_id,
                document_id=passport.document_id,
                page_number=segment.page_number,
                section_title=segment.section_title,
                segment_type="text",
                text=chunk_text,
                entity=passport.entity,
                fiscal_year=passport.fiscal_year,
                scope=passport.scope,
                metadata={
                    "filename": passport.filename,
                    "document_type": passport.document_type,
                    "currency": passport.currency,
                    "units": passport.units
                }
            ))
            
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def _chunk_table(self, passport: DocumentPassport, segment: ParsedSegment, start_idx: int) -> List[IndexedChunk]:
        # If the table headers and rows are present
        if segment.table_headers and segment.table_rows:
            header_line = "| " + " | ".join(segment.table_headers) + " |"
            sep_line = "| " + " | ".join(["---"] * len(segment.table_headers)) + " |"
            
            # Group rows in batches of 5-10 rows to fit within chunk size
            batch_size = 8
            chunks: List[IndexedChunk] = []
            
            for i in range(0, len(segment.table_rows), batch_size):
                row_batch = segment.table_rows[i:i + batch_size]
                row_lines = ["| " + " | ".join(r) + " |" for r in row_batch]
                table_md = "\n".join([header_line, sep_line] + row_lines)
                
                chunk_id = f"{passport.document_id}_p{segment.page_number}_tbl{start_idx + len(chunks):03d}"
                chunks.append(IndexedChunk(
                    chunk_id=chunk_id,
                    document_id=passport.document_id,
                    page_number=segment.page_number,
                    section_title=f"Table: {segment.table_id or 'Financial Table'}",
                    segment_type="table",
                    text=table_md,
                    entity=passport.entity,
                    fiscal_year=passport.fiscal_year,
                    scope=passport.scope,
                    metadata={
                        "filename": passport.filename,
                        "document_type": passport.document_type,
                        "currency": passport.currency,
                        "units": passport.units,
                        "table_id": segment.table_id
                    }
                ))
            return chunks
        else:
            # Fallback to plain text chunking
            return self._chunk_text(passport, segment, start_idx)
