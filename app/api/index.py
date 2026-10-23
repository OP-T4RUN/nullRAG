import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.config import settings
from app.schemas.api_models import IndexResponse
from app.ingestion.parser import DocumentParser
from app.ingestion.passport import PassportExtractor
from app.ingestion.deduplication import DuplicateDetector
from app.ingestion.chunker import StructureAwareChunker
from app.storage.vector_store import vector_storage

router = APIRouter()

parser = DocumentParser()
passport_extractor = PassportExtractor()
duplicate_detector = DuplicateDetector()
chunker = StructureAwareChunker()

@router.post("/index", response_model=IndexResponse)
async def index_documents(
    collection_name: str = Form(..., description="Target collection name"),
    file: Optional[UploadFile] = File(None, description="Zip file containing source documents"),
    directory_path: Optional[str] = Form(None, description="Alternative: local folder path containing documents")
):
    """
    POST /index
    Accepts a zip file of documents (or local folder) and a collection_name.
    Unzips documents, extracts structure, builds Document Passports, 
    detects duplicates, chunks content with provenance, and indexes into Qdrant + BM25.
    """
    if not file and not directory_path:
        raise HTTPException(status_code=400, detail="Either a zip file or a directory_path must be provided.")

    target_dir = settings.DATA_DIR / collection_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unzip uploaded zip file if provided
    if file:
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")
        
        temp_zip_path = target_dir / "uploaded.zip"
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract zip file: {str(e)}")
        finally:
            if temp_zip_path.exists():
                temp_zip_path.unlink()
                
    elif directory_path:
        source_dir = Path(directory_path)
        if not source_dir.exists() or not source_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory path '{directory_path}' does not exist.")
        # Copy files to collection directory
        for item in source_dir.glob("*"):
            if item.is_file():
                shutil.copy(item, target_dir / item.name)

    # 2. Discover documents in target collection directory
    supported_extensions = {".pdf", ".txt", ".md", ".json"}
    document_files = [f for f in target_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]

    if not document_files:
        return IndexResponse(
            collection_name=collection_name,
            status="completed",
            documents_processed=0,
            documents_failed=0,
            chunks_indexed=0,
            message="No supported document files found in zip."
        )

    processed_count = 0
    failed_count = 0
    total_chunks_indexed = 0

    # 3. Process each document
    for doc_path in document_files:
        doc_id = doc_path.stem
        try:
            # Parse segments & tables
            segments = parser.parse_file(doc_path, doc_id)
            if not segments:
                failed_count += 1
                continue

            # Extract Document Passport
            passport = passport_extractor.extract_passport(
                collection_name=collection_name,
                document_id=doc_id,
                filename=doc_path.name,
                segments=segments
            )

            # Check duplicates
            rel_type, related_id = duplicate_detector.check_relationship(passport)
            if rel_type == "exact_duplicate":
                # Skip duplicate chunks to avoid index bloat
                continue

            # Save normalized parsed output for inspection and caching
            parsed_out_dir = settings.STORAGE_DIR / "parsed" / collection_name
            parsed_out_dir.mkdir(parents=True, exist_ok=True)
            with open(parsed_out_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "passport": passport.model_dump(),
                    "segments": [s.model_dump() for s in segments]
                }, f, indent=2)

            # Extract & persist open-world metadata assertions in SQLite
            try:
                from app.knowledge import assertion_extractor, assertion_store, provenance_manager, RelationshipType
                assertions = assertion_extractor.extract_assertions(passport, segments)
                assertion_store.save_assertions(assertions)

                if related_id:
                    rel_type_enum = RelationshipType.SAME_REPORT_FAMILY if "family" in rel_type else RelationshipType.PREVIOUS_VERSION_OF
                    provenance_manager.link_document_family(
                        doc_id, related_id, rel_type_enum, {"detection": rel_type}
                    )
            except Exception:
                pass

            duplicate_detector.register_passport(passport)

            # Structure-aware chunking
            chunks = chunker.chunk_document(passport, segments)
            
            # Index in Qdrant + BM25
            vector_storage.index_chunks(collection_name, chunks)
            
            processed_count += 1
            total_chunks_indexed += len(chunks)

        except Exception as e:
            failed_count += 1
            continue

    return IndexResponse(
        collection_name=collection_name,
        status="completed",
        documents_processed=processed_count,
        documents_failed=failed_count,
        chunks_indexed=total_chunks_indexed,
        message=f"Successfully indexed {processed_count} documents ({total_chunks_indexed} chunks)."
    )
