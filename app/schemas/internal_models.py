from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ParsedSegment(BaseModel):
    segment_id: str
    document_id: str
    page_number: int
    section_title: Optional[str] = None
    segment_type: str = "text"  # "text", "table", "heading"
    text: str
    table_id: Optional[str] = None
    table_headers: Optional[List[str]] = None
    table_rows: Optional[List[List[str]]] = None
    source_location: Optional[Dict[str, Any]] = None
    content_hash: str

class DocumentPassport(BaseModel):
    document_id: str
    collection_name: str
    filename: str
    entity: Optional[str] = None
    fiscal_year: Optional[str] = None
    publication_date: Optional[str] = None
    document_type: Optional[str] = None  # e.g., "annual_report", "risk_disclosure", "quarterly_report"
    scope: Optional[str] = None          # e.g., "consolidated", "subsidiary", "business_unit"
    version: Optional[str] = None
    currency: Optional[str] = None
    units: Optional[str] = None
    content_hash: str
    page_count: int = 1
    metadata_confidence: float = 1.0
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)

class IndexedChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    section_title: Optional[str] = None
    segment_type: str = "text"
    text: str
    entity: Optional[str] = None
    fiscal_year: Optional[str] = None
    scope: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvidenceObligation(BaseModel):
    obligation_id: str
    description: str
    required_evidence_type: str = "general" # "numeric_fact", "comparison", "explanation", "mitigation", "definition"
    required_entity: Optional[str] = None
    required_period: Optional[str] = None
    required_scope: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    coverage_status: str = "uncovered"     # "uncovered", "partially_covered", "fully_covered"
    extracted_facts: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_attempts: int = 0
