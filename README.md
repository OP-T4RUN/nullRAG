# nullRAG: Smart Knowledge Base for Agents

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Storage-red.svg?logo=qdrant)](https://qdrant.tech)
[![Langfuse](https://img.shields.io/badge/Observability-Langfuse-black.svg)](https://langfuse.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)](https://python.org)

**nullRAG** is an evidence-aware knowledge retrieval and grounded question-answering system designed for high-stakes business and financial research copilots.

---

## 1. Problem Addressed & Why Standard RAG Fails

Standard retrieval pipelines (chunk -> embed -> vector search -> top-k stuffing -> rerank) plateau at ~0.6 precision and recall due to two fundamental failure modes:

1. **Failure Mode 1: Near-Duplicate / Highly Similar Documents**
   - Documents covering different fiscal years (e.g., FY2024 vs. FY2025) or scopes (consolidated group vs. single subsidiary) share identical financial vocabulary. Vector similarity conflates them.
2. **Failure Mode 2: Chunk-Size Dilemma & Incomplete Retrieval**
   - Answers spread across 20-30 pages get missed by fixed top-k search. Increasing chunk size or top-k bloats context with noise and spikes latency/cost without ensuring evidence completeness.

---

## 2. Core Architectural Innovations

### A. Document Passport & Scope Identity
- Every indexed document receives an immutable Passport extracting **Entity, Fiscal Year/Period, Scope (Consolidated vs. Subsidiary), Document Type, Units/Currency**, SHA-256 hash, and MinHash signature.
- Strict period/scope filtering prevents semantically similar but chronologically wrong evidence from leaking into the context.

### B. MinHash & Document Family Mapping
- Detects exact clones and near-duplicate revisions while strictly preserving distinct reporting years and entity scopes.

### C. Evidence Obligation Ledger & Bounded Gap Filling
- Rather than asking *"Which chunks look most similar?"*, the query planner constructs testable evidence requirements:
  - *Obligation 1: FY2024 metric + unit + source citation*
  - *Obligation 2: FY2025 metric + unit + source citation*
  - *Obligation 3: Scope compatibility*
  - *Obligation 4: Reported reason/explanation for change*
  - *Obligation 5: Mitigation measures*
- An **Evidence Coverage Ledger** audits retrieved candidates against each obligation. If an obligation is missing (e.g., missing FY2024 evidence in a comparison query), a targeted secondary retrieval is dispatched specifically for the gap.

### D. Grounded Answer Synthesis & Page Provenance
- Preserves document IDs and exact page numbers from PDF extraction (`pdfplumber` / `pypdf`).
- Enforces real document/page citations in brackets `[Document_ID, Page X]` and flags missing information rather than hallucinating answers.

### E. Complete Traceability & Observability
- Every `/query` response outputs a structured JSON `trace` array (`step_name`, `description`, `input`, `output`).
- Automatic span logging to **Langfuse** for latency, token consumption, and cost tracking.

---

## 3. API Contract

### `POST /index`
Indexes a zip file of documents (or local folder) into a named collection.

- **Request** (`multipart/form-data`):
  - `collection_name` (string, required): Name of the collection.
  - `file` (binary zip file, optional): Zip archive of PDF/text documents.
  - `directory_path` (string, optional): Server path to document folder.
- **Response**:
  ```json
  {
    "collection_name": "financial_reports",
    "status": "completed",
    "documents_processed": 50,
    "documents_failed": 0,
    "chunks_indexed": 1240,
    "message": "Successfully indexed 50 documents (1240 chunks)."
  }
  ```

### `POST /query`
Performs evidence obligation planning, hybrid retrieval, cross-eligibility reranking, targeted gap filling, and grounded answer synthesis.

- **Request** (`application/json`):
  ```json
  {
    "collection_name": "financial_reports",
    "query": "Compare credit risk exposure between FY2024 and FY2025 and explain the major changes."
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Based on verified source documents: [acme_fy2024, Page 1] reported exposure of $120 million for FY2024...",
    "chunks": [
      {
        "document_id": "acme_fy2024",
        "page_number": 1,
        "chunk_id": "acme_fy2024_p1_c001",
        "text": "Credit risk exposure was reported as $120 million for FY2024...",
        "metadata": {
          "fiscal_year": "FY2024",
          "scope": "consolidated"
        }
      }
    ],
    "trace": [
      {
        "step_name": "query_planning",
        "description": "Decomposed query into explicit entity, fiscal-year, scope, and evidence obligations",
        "input": { "query": "..." },
        "output": { "obligations": [...] }
      },
      {
        "step_name": "similarity_search",
        "description": "Executed hybrid dense and BM25 search with Reciprocal Rank Fusion",
        "input": { ... },
        "output": { ... }
      },
      {
        "step_name": "rerank",
        "description": "Applied cross-eligibility checks (period/scope constraints) and reranked candidate pool",
        "input": { ... },
        "output": { ... }
      },
      {
        "step_name": "evidence_coverage_ledger",
        "description": "Checked obligation coverage status and executed targeted queries for missing evidence",
        "input": { ... },
        "output": { ... }
      },
      {
        "step_name": "llm_answer_generation",
        "description": "Generated final grounded response enforcing document and page citations",
        "input": { ... },
        "output": { ... }
      }
    ]
  }
  ```

---

## 4. Setup & Running

### Option A: Local Python Environment
```bash
# 1. Activate virtual environment
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your LLM_API_KEY (OpenCode/OpenAI) and LANGFUSE credentials

# 4. Run tests
python -m pytest -v

# 5. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Docker Compose
```bash
docker compose up --build -d
```

---

## 5. Running Benchmark Evaluation

To evaluate on the 100 Q&A benchmark:
```bash
python scripts/run_evaluation.py --benchmark_file data/benchmark_qa.json --collection_name financial_reports
```
