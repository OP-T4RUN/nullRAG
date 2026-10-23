import asyncio
import json
from app.schemas.api_models import QueryRequest
from app.api.query import query_knowledge_base

async def main():
    print("Testing EFAR end-to-end query on 'train_test_collection'...")
    req = QueryRequest(
        collection_name="train_test_collection",
        query="What are the regulatory requirements and adjustments for credit valuation under BCBS guidelines?",
        options={"candidate_depth": 10, "final_context_k": 5}
    )
    
    response = await query_knowledge_base(req)
    print("\n--- Answer ---")
    print(response.answer[:500] + ("..." if len(response.answer) > 500 else ""))
    print(f"\nTotal Citations: {len(response.chunks)}")
    for c in response.chunks[:3]:
        print(f" - [{c.document_id}, Page {c.page_number}] (Chunk: {c.chunk_id})")
    
    print(f"\nTotal Trace Steps: {len(response.trace)}")
    for step in response.trace:
        print(f" - Step '{step.step_name}': {step.description}")

if __name__ == "__main__":
    asyncio.run(main())
