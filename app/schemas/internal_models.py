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
    requirement_type: str = "general" # "fact", "comparison", "explanation", "mitigation", "definition", "qualification"
    required_entity: Optional[str] = None
    required_period: Optional[str] = None
    required_scope: Optional[str] = None
    required_concepts: List[str] = Field(default_factory=list)
    hard_constraints: Dict[str, Any] = Field(default_factory=dict)
    soft_preferences: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    required_support: int = 1
    keywords: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    coverage_status: str = "uncovered"     # "uncovered", "partially_covered", "fully_covered"
    status: str = "unresolved"             # "supported", "conflicted", "unresolved"
    extracted_facts: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_attempts: int = 0

    @property
    def required_evidence_type(self) -> str:
        return self.requirement_type

    @required_evidence_type.setter
    def required_evidence_type(self, value: str):
        self.requirement_type = value

class QueryPlan(BaseModel):
    query_id: str
    original_query: str
    intent: str = "informational"          # "factual", "comparison", "multi_part", "investigation"
    entity_constraints: List[str] = Field(default_factory=list)
    scope_constraints: Optional[str] = None
    period_constraints: List[str] = Field(default_factory=list)
    obligations: List[EvidenceObligation] = Field(default_factory=list)
    retrieval_budget: Dict[str, Any] = Field(default_factory=lambda: {
        "candidate_depth": 25,
        "max_context_chunks": 10,
        "max_recovery_rounds": 1
    })
