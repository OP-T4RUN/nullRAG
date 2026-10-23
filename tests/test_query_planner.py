from app.retrieval.query_planner import QueryPlanner

def test_query_planner_cross_year_comparison():
    planner = QueryPlanner()
    query = "Compare the credit risk exposure between FY2024 and FY2025 and explain the major changes."
    meta, obligations = planner.plan_query(query)

    assert meta["is_comparison"] is True
    assert meta["needs_explanation"] is True
    assert "FY2024" in meta["years_detected"]
    assert "FY2025" in meta["years_detected"]

    # Check obligations
    obl_ids = [o.obligation_id for o in obligations]
    assert "obl_period_FY2024" in obl_ids
    assert "obl_period_FY2025" in obl_ids
    assert "obl_delta_comparison" in obl_ids
    assert "obl_explanation" in obl_ids

def test_query_planner_single_period_mitigation():
    planner = QueryPlanner()
    query = "What risk mitigation measures were reported for consolidated credit loss in FY2025?"
    meta, obligations = planner.plan_query(query)

    assert meta["needs_mitigation"] is True
    assert meta["scope_detected"] == "consolidated"
    assert "FY2025" in meta["years_detected"]
    
    obl_types = [o.required_evidence_type for o in obligations]
    assert "mitigation" in obl_types
