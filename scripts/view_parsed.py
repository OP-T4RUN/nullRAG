import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.storage.vector_store import vector_storage

def view_parsed_file(parsed_file_path: Path, max_segments: int = 5):
    if not parsed_file_path.exists():
        print(f"File not found: {parsed_file_path}")
        return

    with open(parsed_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print(f"PARSED DOCUMENT DETAILS: {parsed_file_path.name}")
    print("=" * 80)

    # 1. Passport
    passport = data.get("passport", {})
    print("\n[Document Passport]")
    for k, v in passport.items():
        if k != "extracted_metadata":
            print(f"  {k:20}: {v}")

    # 2. Segments & Tables
    segments = data.get("segments", [])
    print(f"\n[Segments Summary]")
    print(f"  Total segments: {len(segments)}")
    
    tables = [s for s in segments if s.get("segment_type") == "table"]
    texts = [s for s in segments if s.get("segment_type") == "text"]
    print(f"  - Text pages  : {len(texts)}")
    print(f"  - Tables      : {len(tables)}")

    if tables:
        print("\n" + "-" * 40)
        print("SAMPLE EXTRACTED TABLE:")
        print("-" * 40)
        print(f"Page: {tables[0].get('page_number')} | Table ID: {tables[0].get('table_id')}")
        print(tables[0].get("text"))

    print("\n" + "-" * 40)
    print(f"FIRST {min(max_segments, len(segments))} SEGMENTS PREVIEW:")
    print("-" * 40)
    for i, seg in enumerate(segments[:max_segments]):
        seg_type = seg.get("segment_type", "text")
        page = seg.get("page_number")
        text = seg.get("text", "").replace("\n", " ")
        snippet = text[:150] + ("..." if len(text) > 150 else "")
        print(f"[{i+1}] (Page {page} | Type: {seg_type}): {snippet}")

def view_qdrant_points(collection_name: str, limit: int = 5):
    print("=" * 80)
    print(f"QDRANT STORED POINTS: collection='{collection_name}'")
    print("=" * 80)

    try:
        col = vector_storage.client.get_collection(collection_name)
        print(f"Collection status : {col.status}")
        print(f"Total points stored: {col.points_count}")
        print(f"Vector dimension  : {col.config.params.vectors.size}")
        print(f"Distance metric   : {col.config.params.vectors.distance}")
    except Exception as e:
        print(f"Could not inspect collection '{collection_name}': {e}")
        return

    # Scroll points
    try:
        points, _ = vector_storage.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        print(f"\nSample {len(points)} Stored Points (with Payloads):")
        for idx, pt in enumerate(points):
            payload = pt.payload or {}
            print(f"\nPoint #{idx+1} [Qdrant ID: {pt.id}]")
            print(f"  - chunk_id    : {payload.get('chunk_id')}")
            print(f"  - document_id : {payload.get('document_id')}")
            print(f"  - page_number : {payload.get('page_number')}")
            print(f"  - segment_type: {payload.get('segment_type')}")
            print(f"  - fiscal_year : {payload.get('fiscal_year')}")
            print(f"  - scope       : {payload.get('scope')}")
            print(f"  - text snippet: {payload.get('text', '')[:160]}...")
    except Exception as e:
        print(f"Error scrolling points: {e}")

def main():
    parser = argparse.ArgumentParser(description="View parsed documents and Qdrant storage")
    parser.add_argument("--collection", type=str, default="train_test_collection", help="Collection name to inspect in Qdrant")
    parser.add_argument("--doc", type=str, default=None, help="Document ID or path to JSON parsed document")
    parser.add_argument("--qdrant", action="store_true", help="View raw Qdrant stored points and payloads")
    args = parser.parse_args()

    parsed_dir = settings.STORAGE_DIR / "parsed" / args.collection
    
    if args.doc:
        doc_path = Path(args.doc)
        if not doc_path.exists() and not doc_path.suffix:
            doc_path = parsed_dir / f"{args.doc}.json"
        view_parsed_file(doc_path)
    elif args.qdrant or not parsed_dir.exists():
        view_qdrant_points(args.collection)
    else:
        # List available parsed docs
        json_files = list(parsed_dir.glob("*.json"))
        if json_files:
            print(f"Found {len(json_files)} parsed documents in {parsed_dir}:")
            for jf in json_files:
                print(f" - {jf.stem} ({jf.name})")
            print("\nViewing first document:")
            view_parsed_file(json_files[0])
        else:
            view_qdrant_points(args.collection)

if __name__ == "__main__":
    main()
