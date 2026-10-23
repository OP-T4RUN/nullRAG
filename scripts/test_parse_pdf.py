import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.parser import DocumentParser
from app.ingestion.passport import PassportExtractor
from app.ingestion.chunker import StructureAwareChunker
from app.storage.vector_store import vector_storage

def test_single_pdf(pdf_path_str: str, collection_name: str = "train_test_collection"):
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"Error: PDF file '{pdf_path}' not found.")
        return

    print("=" * 80)
    print(f"1. Parsing PDF: {pdf_path.name}")
    print(f"   File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    print("=" * 80)

    # 1. Parse Segments & Tables
    parser = DocumentParser()
    doc_id = pdf_path.stem
    segments = parser.parse_file(pdf_path, doc_id)

    text_segments = [s for s in segments if s.segment_type == "text"]
    table_segments = [s for s in segments if s.segment_type == "table"]

    print(f"\n[Extraction Summary]")
    print(f" - Total extracted segments: {len(segments)}")
    print(f" - Text page segments: {len(text_segments)}")
    print(f" - Structured table segments: {len(table_segments)}")

    # 2. Extract Document Passport
    passport_extractor = PassportExtractor()
    passport = passport_extractor.extract_passport(
        collection_name=collection_name,
        document_id=doc_id,
        filename=pdf_path.name,
        segments=segments
    )

    # Save normalized parsed output for inspection and caching
    parsed_out_dir = settings.STORAGE_DIR / "parsed" / collection_name
    parsed_out_dir.mkdir(parents=True, exist_ok=True)
    parsed_json_path = parsed_out_dir / f"{doc_id}.json"
    
    with open(parsed_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "passport": passport.model_dump(),
            "segments": [s.model_dump() for s in segments]
        }, f, indent=2)
    print(f" - Saved normalized parsed JSON to: {parsed_json_path}")

    print(f"\n[Document Passport]")
    print(f" - Document ID: {passport.document_id}")
    print(f" - Filename: {passport.filename}")
    print(f" - Entity: {passport.entity}")
    print(f" - Fiscal Year / Period: {passport.fiscal_year}")
    print(f" - Document Type: {passport.document_type}")
    print(f" - Scope: {passport.scope}")
    print(f" - Currency: {passport.currency}")
    print(f" - Units: {passport.units}")
    print(f" - Total Pages: {passport.page_count}")
    print(f" - Content Hash (SHA256): {passport.content_hash[:16]}...")
    print(f" - Metadata Confidence: {passport.metadata_confidence:.2f}")

    # 3. Structure-aware Chunking
    chunker = StructureAwareChunker(chunk_size=350, chunk_overlap=40)
    chunks = chunker.chunk_document(passport, segments)
    print(f"\n[Chunking Summary]")
    print(f" - Total chunks generated: {len(chunks)}")

    # Preview Tables if any
    if table_segments:
        print(f"\n[Sample Extracted Table (Page {table_segments[0].page_number})]")
        lines = table_segments[0].text.split("\n")
        print("\n".join(lines[:8]))
        if len(lines) > 8:
            print(f"  ... ({len(lines) - 8} more rows)")

    # Preview First 2 Chunks
    print(f"\n[Sample Chunk 1 (ID: {chunks[0].chunk_id})]")
    print(f" Page: {chunks[0].page_number} | Type: {chunks[0].segment_type}")
    print(f" Text preview:\n{chunks[0].text[:300]}...")

    if len(chunks) > 1:
        print(f"\n[Sample Chunk 2 (ID: {chunks[1].chunk_id})]")
        print(f" Page: {chunks[1].page_number} | Type: {chunks[1].segment_type}")
        print(f" Text preview:\n{chunks[1].text[:300]}...")

    # 4. Save to Qdrant Vector Storage + BM25
    print("\n" + "=" * 80)
    print(f"2. Saving {len(chunks)} chunks & dense vectors to Qdrant collection '{collection_name}'...")
    print("=" * 80)

    vector_storage.index_chunks(collection_name, chunks)
    
    # 5. Verify Qdrant Storage
    qdrant_col_info = vector_storage.client.get_collection(collection_name)
    points_count = qdrant_col_info.points_count
    print(f"[Qdrant Verification]")
    print(f" - Collection '{collection_name}' points count: {points_count}")
    print(f" - Vectors configuration: {qdrant_col_info.config.params.vectors}")

    # 6. Test a retrieval search against Qdrant
    test_query = "credit valuation adjustment risk framework"
    print(f"\n[Verification Search from Qdrant: query='{test_query}']")
    hits = vector_storage.hybrid_search(collection_name, test_query, limit=3)
    for idx, hit in enumerate(hits):
        print(f"\n Result #{idx+1} [RRF Score: {hit.get('rrf_score', 0):.4f}]")
        print(f" - Chunk ID: {hit.get('chunk_id')}")
        print(f" - Source Doc: {hit.get('document_id')}, Page: {hit.get('page_number')}")
        print(f" - Content Snippet: {hit.get('text', '')[:200]}...")

    print("\n" + "=" * 80)
    print("SUCCESS: PDF parsed, structured, chunked, and stored in Qdrant!")
    print("=" * 80)

if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else r"D:\workspace\nullRAG\Train\Documents\BCBS-CreditValuation.pdf"
    test_single_pdf(pdf_file)
