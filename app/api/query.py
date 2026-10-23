import time
from fastapi import APIRouter, HTTPException

from app.schemas.api_models import QueryRequest, QueryResponse
from app.observability.tracing import ExecutionTracer
from app.retrieval.query_planner import query_planner
from app.retrieval.reranker import EvidenceEligibilityReranker
from app.retrieval.support_matrix import support_matrix_builder
from app.retrieval.selector import coverage_selector
from app.retrieval.recovery import recovery_engine
from app.retrieval.consistency import consistency_analyzer
from app.generation.answer import AnswerGenerator
from app.storage.vector_store import vector_storage

router = APIRouter()

reranker = EvidenceEligibilityReranker()
answer_generator = AnswerGenerator()

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    POST /query
    EFAR-powered query execution pipeline:
    1. Query Compiler & Evidence Obligation Generation
    2. Hybrid Candidate Retrieval (Dense + BM25 with RRF)
    3. Evidence Support Matrix Assessment
    4. Coverage-Constrained Evidence Selection
    5. Bounded Gap Recovery
    6. Consistency & Scope Analysis
    7. Grounded Answer Synthesis with Claim Validation
    """
    collection_name = request.collection_name
    query = request.query
    options = request.options or {}

    tracer = ExecutionTracer(
        trace_name="efar_query_pipeline",
        collection_name=collection_name,
        query=query
    )

    # Step 1: Query Compilation & Evidence Obligation Generation
    t0 = time.time()
    plan = query_planner.compile_plan(query)
    obligations = plan.obligations
    t_planning = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="query_planning",
        description="EFAR Query Compiler: decomposed query into structured intent, hard constraints, and testable obligations",
        step_input={"query": query},
        step_output={
            "query_id": plan.query_id,
            "intent": plan.intent,
            "period_constraints": plan.period_constraints,
            "entity_constraints": plan.entity_constraints,
            "scope_constraints": plan.scope_constraints,
            "obligations": [o.model_dump() for o in obligations]
        },
        duration_ms=t_planning
    )

    # Step 2: Hybrid Candidate Retrieval (Dense + BM25 with RRF)
    t0 = time.time()
    candidate_depth = options.get("candidate_depth", plan.retrieval_budget.get("candidate_depth", 25))
    candidates = vector_storage.hybrid_search(
        collection_name=collection_name,
        query=query,
        limit=candidate_depth
    )
    t_retrieval = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="similarity_search",
        description="Executed hybrid dense and BM25 search with Reciprocal Rank Fusion",
        step_input={"query": query, "candidate_depth": candidate_depth},
        step_output={
            "candidates_retrieved": len(candidates),
            "top_candidates": [{"chunk_id": c["chunk_id"], "page": c["page_number"], "year": c.get("fiscal_year")} for c in candidates[:5]]
        },
        duration_ms=t_retrieval
    )

    # Step 3: Evidence Support Matrix
    t0 = time.time()
    support_matrix = support_matrix_builder.build_matrix(candidates, obligations)
    t_matrix = (time.time() - t0) * 1000

    matrix_summary = {}
    for o in obligations:
        assessments = support_matrix.get(o.obligation_id, {})
        matrix_summary[o.obligation_id] = {
            "direct": sum(1 for a in assessments.values() if a.relation.value == "direct_support"),
            "partial": sum(1 for a in assessments.values() if a.relation.value == "partial_support"),
            "ineligible": sum(1 for a in assessments.values() if a.failed_constraints)
        }

    tracer.add_step(
        step_name="support_matrix",
        description="EFAR Support Matrix: evaluated candidate-to-obligation support strength and hard constraint compliance",
        step_input={"candidates_count": len(candidates), "obligations_count": len(obligations)},
        step_output=matrix_summary,
        duration_ms=t_matrix
    )

    # Step 4: Coverage-Constrained Evidence Selection
    t0 = time.time()
    max_context_chunks = options.get("final_context_k", plan.retrieval_budget.get("max_context_chunks", 10))
    selected_chunks, evaluated_obligations, selection_summary = coverage_selector.select_evidence(
        candidates=candidates,
        obligations=obligations,
        support_matrix=support_matrix,
        max_context_chunks=max_context_chunks
    )
    t_selection = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="evidence_selection",
        description="Coverage-constrained selector: greedily selected non-redundant evidence maximizing marginal obligation coverage",
        step_input={"candidates_pool": len(candidates), "max_budget": max_context_chunks},
        step_output=selection_summary,
        duration_ms=t_selection
    )

    # Step 5: Bounded Gap Recovery
    t0 = time.time()
    unresolved = [o for o in evaluated_obligations if o.status == "unresolved"]
    if unresolved:
        completed_chunks, final_obligations, recovery_report = recovery_engine.recover_missing_evidence(
            collection_name=collection_name,
            base_query=query,
            unresolved_obligations=unresolved,
            current_selected_chunks=selected_chunks,
            max_recovery_rounds=1
        )
    else:
        completed_chunks = selected_chunks
        final_obligations = evaluated_obligations
        recovery_report = {"rounds_run": 0, "recovered_chunks": 0, "status": "all_obligations_covered"}
    t_recovery = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="gap_recovery",
        description="EFAR Bounded Recovery Engine: triggered focused secondary retrieval for unresolved obligations",
        step_input={"unresolved_count": len(unresolved)},
        step_output=recovery_report,
        duration_ms=t_recovery
    )

    # Step 6: Consistency & Scope Analysis
    t0 = time.time()
    consistency_report = consistency_analyzer.analyze_consistency(completed_chunks)
    t_consistency = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="consistency_analysis",
        description="Audited evidence for accounting scope (consolidated vs standalone) and reporting period distinctions",
        step_input={"selected_chunks_count": len(completed_chunks)},
        step_output=consistency_report,
        duration_ms=t_consistency
    )

    # Step 7: Grounded Answer Generation & Citation Audit
    t0 = time.time()
    answer, citations, usage = await answer_generator.generate_answer(
        query=query,
        evidence_chunks=completed_chunks,
        obligations=final_obligations,
        consistency_report=consistency_report
    )
    t_gen = (time.time() - t0) * 1000

    tracer.add_step(
        step_name="llm_answer_generation",
        description="Generated grounded response enforcing document/page citations and executed claim validation",
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
