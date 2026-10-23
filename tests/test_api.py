import io
import zipfile
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_index_and_query_flow():
    # 1. Create in-memory zip archive with test documents
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "doc_fy2024.txt",
            "Acme Global Inc. Consolidated Annual Report FY2024.\n"
            "Credit risk exposure was reported as $120 million for FY2024.\n"
            "The company managed credit risk through selective underwriting."
        )
        zf.writestr(
            "doc_fy2025.txt",
            "Acme Global Inc. Consolidated Annual Report FY2025.\n"
            "Credit risk exposure was reported as $150 million for FY2025.\n"
            "The major increase of $30 million was driven by expansion in commercial real estate lending.\n"
            "Mitigation measures include higher collateral requirements and portfolio stress testing."
        )
    zip_buffer.seek(0)

    # 2. Call POST /index
    index_res = client.post(
        "/index",
        data={"collection_name": "test_financial_reports"},
        files={"file": ("reports.zip", zip_buffer, "application/zip")}
    )
    assert index_res.status_code == 200
    idx_data = index_res.json()
    assert idx_data["status"] == "completed"
    assert idx_data["documents_processed"] >= 1
    assert idx_data["chunks_indexed"] >= 1

    # 3. Call POST /query
    query_payload = {
        "collection_name": "test_financial_reports",
        "query": "Compare credit risk exposure between FY2024 and FY2025 and explain the major changes."
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()

    assert "answer" in q_data
    assert len(q_data["chunks"]) > 0
    assert len(q_data["trace"]) >= 4

    # Verify trace steps structure matches hackathon requirement
    for step in q_data["trace"]:
        assert "step_name" in step
        assert "description" in step
        assert "input" in step
        assert "output" in step

    # Verify citation fields
    citation = q_data["chunks"][0]
    assert "document_id" in citation
    assert "page_number" in citation
    assert "chunk_id" in citation
