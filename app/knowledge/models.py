from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class AssertionStatus(str, Enum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    UNRESOLVED = "unresolved"

class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    CONTRADICTS = "contradicts"
    NEAR_DUPLICATE_OF = "near_duplicate_of"
    PREVIOUS_VERSION_OF = "previous_version_of"
    SAME_REPORT_FAMILY = "same_report_family"

class MetadataAssertion(BaseModel):
    assertion_id: str
    document_id: str
    raw_attribute_name: str
    canonical_concept: Optional[str] = None
    value: Any
    value_type: str = "string"  # "string", "float", "int", "boolean", "dict", "list"
    unit: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list, description="Chunk IDs or segment IDs providing provenance")
    confidence: Optional[float] = 1.0
    status: AssertionStatus = AssertionStatus.SUPPORTED
    extractor_version: str = "1.0.0"

class EvidenceRelationship(BaseModel):
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentQualitySignal(BaseModel):
    document_id: str
    page_number: int
    extraction_status: str = "success"  # "success", "partial", "failed"
    has_usable_text: bool = True
    table_extraction_status: str = "none"  # "none", "extracted", "ocr_needed", "malformed"
    ocr_used: bool = False
    parser_version: str = "1.0.0"
    quality_warnings: List[str] = Field(default_factory=list)
