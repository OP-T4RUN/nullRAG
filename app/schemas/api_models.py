from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class TraceStep(BaseModel):
    step_name: str = Field(..., description="Short identifier for the step (e.g. metadata_filter, similarity_search, rerank, llm_answer_generation)")
    description: str = Field(..., description="What that step was doing")
    input: Any = Field(..., description="What went into that step")
    output: Any = Field(..., description="What came out of that step")

class ChunkCitation(BaseModel):
    document_id: str
    page_number: int
    chunk_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IndexResponse(BaseModel):
    collection_name: str
    status: str
    documents_processed: int
    documents_failed: int
    chunks_indexed: int
    message: Optional[str] = None

class QueryRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection to query")
    query: str = Field(..., description="User query / question")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional retrieval tuning options")

class QueryResponse(BaseModel):
    answer: str
    chunks: List[ChunkCitation]
    trace: List[TraceStep]
