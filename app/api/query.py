import time
from fastapi import APIRouter, HTTPException

from app.schemas.api_models import QueryRequest, QueryResponse
from app.observability.tracing import ExecutionTracer
from app.retrieval.query_planner import QueryPlanner
from app.retrieval.reranker import EvidenceEligibilityReranker
from app.retrieval.coverage_ledger import EvidenceCoverageLedger
from app.generation.answer import AnswerGenerator
from app.storage.vector_store import vector_storage

router = APIRouter()

query_planner = QueryPlanner()
reranker = EvidenceEligibilityReranker()
coverage_ledger = EvidenceCoverageLedger()
answer_generator = AnswerGenerator()

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    POST /query
    Retrieves evidence, fills missing obligations, generates grounded answer,
    and returns document/page citations + step-by-step trace.
    """
    collection_name = request.collection_name
    query = request.query
    options = request.options or {}

    tracer = ExecutionTracer(
        trace_name="query_pipeline",
        collection_name=collection_name,
        query=query
    )

    # Step 1: Query Understanding & Evidence Obligation Generation
    t0 = time.time()
    query_meta, obligations = query_planner.plan_query(query)
    t_planning = (time.time() - t0) * 1000
    
    tracer.add_step(
        step_name="query_planning",
        description="Decomposed query into explicit entity, fiscal-year, scope, and evidence obligations",
        step_input={"query": query},
        step_output={
            "query_meta": query_meta,
            "obligations": [o.model_dump() for o in obligations]
        },
        duration_ms=t_planning
    )

    # Step 2: Hybrid Candidate Retrieval (Dense + BM25 with RRF)
    t0 = time.time()
    candidate_depth = options.get("candidate_depth", 25)
    candidates = vector_storage.hybrid_search(
        collection_name=collection_name,
        query=query,
        limit=candidate_depth
    )
    t_retrieval = (time.time() - t0) * 1000
    
    tracer.add_step(
        step_name="similarity_search",
        description="Executed hybrid dense and BM25 search with Reciprocal Rank Fusion",
        step_input={"query": query, "depth": candidate_depth},
        step_output={"candidates_retrieved": len(candidates), "top_chunk_ids": [c["chunk_id"] for c in candidates[:5]]},
        duration_ms=t_retrieval
    )

    # Step 3: Evidence Eligibility Check & Reranking
    t0 = time.time()
    final_k = options.get("final_context_k", 8)
    reranked_chunks = reranker.rerank(
        query=query,
        candidates=candidates,
        obligations=obligations,
        limit=final_k
    )
    t_rerank = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="rerank",
        description="Applied cross-eligibility checks (period/scope constraints) and reranked candidate pool",
        step_input={"candidates_count": len(candidates)},
        step_output={
            "selected_count": len(reranked_chunks),
            "selected_chunk_ids": [c["chunk_id"] for c in reranked_chunks]
        },
        duration_ms=t_rerank
    )

    # Step 4: Evidence Coverage Evaluation & Targeted Gap Filling
    t0 = time.time()
    evaluated_obligations = coverage_ledger.evaluate_coverage(obligations, reranked_chunks)
    completed_chunks, final_obligations = coverage_ledger.fill_gaps(
        collection_name=collection_name,
        obligations=evaluated_obligations,
        current_candidates=reranked_chunks,
        base_query=query,
        max_gap_queries=2
    )
    t_coverage = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="evidence_coverage_ledger",
        description="Checked obligation coverage status and executed targeted queries for missing evidence",
        step_input={"initial_obligations": [o.coverage_status for o in evaluated_obligations]},
        step_output={
            "final_obligations": [{o.obligation_id: o.coverage_status} for o in final_obligations],
            "total_evidence_chunks": len(completed_chunks)
        },
        duration_ms=t_coverage
    )

    # Step 5: Grounded Answer Generation & Citation Audit
    t0 = time.time()
    answer, citations, usage = await answer_generator.generate_answer(
        query=query,
        evidence_chunks=completed_chunks,
        obligations=final_obligations
    )
    t_gen = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="llm_answer_generation",
        description="Generated final grounded response enforcing document and page citations",
        step_input={"evidence_chunks_count": len(completed_chunks), "query": query},
        step_output={
            "answer_preview": answer[:250],
            "citations_count": len(citations),
            "usage": usage
        },
        duration_ms=t_gen
    )

    # Finalize trace
    steps = tracer.finalize(
        answer=answer,
        total_tokens=usage.get("total_tokens", 0),
        cost_usd=usage.get("cost_usd", 0.0)
    )

    return QueryResponse(
        answer=answer,
        chunks=citations,
        trace=steps
    )
