import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import settings
from app.knowledge.models import MetadataAssertion, AssertionStatus, EvidenceRelationship, RelationshipType
from app.knowledge.registry import concept_registry
from app.schemas.internal_models import ParsedSegment, DocumentPassport, IndexedChunk

class AssertionStore:
    """
    SQLite-backed persistence layer for open-world metadata assertions and relationships.
    Does not require database schema alterations when novel attributes appear.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AssertionStore, cls).__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        self.db_path = settings.STORAGE_DIR / "knowledge.sqlite"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table 1: Metadata assertions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    raw_attribute_name TEXT NOT NULL,
                    canonical_concept TEXT,
                    value_json TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    unit TEXT,
                    evidence_refs_json TEXT NOT NULL,
                    confidence REAL,
                    status TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assertions_doc ON metadata_assertions(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assertions_concept ON metadata_assertions(canonical_concept)")

            # Table 2: Evidence relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evidence_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON evidence_relationships(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON evidence_relationships(target_id)")
            conn.commit()

    def save_assertion(self, assertion: MetadataAssertion):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata_assertions (
                    assertion_id, document_id, raw_attribute_name, canonical_concept,
                    value_json, value_type, unit, evidence_refs_json,
                    confidence, status, extractor_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assertion.assertion_id,
                assertion.document_id,
                assertion.raw_attribute_name,
                assertion.canonical_concept,
                json.dumps(assertion.value),
                assertion.value_type,
                assertion.unit,
                json.dumps(assertion.evidence_refs),
                assertion.confidence,
                assertion.status.value if isinstance(assertion.status, AssertionStatus) else str(assertion.status),
                assertion.extractor_version
            ))
            conn.commit()

    def save_assertions(self, assertions: List[MetadataAssertion]):
        if not assertions:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            records = [
                (
                    a.assertion_id,
                    a.document_id,
                    a.raw_attribute_name,
                    a.canonical_concept,
                    json.dumps(a.value),
                    a.value_type,
                    a.unit,
                    json.dumps(a.evidence_refs),
                    a.confidence,
                    a.status.value if isinstance(a.status, AssertionStatus) else str(a.status),
                    a.extractor_version
                )
                for a in assertions
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO metadata_assertions (
                    assertion_id, document_id, raw_attribute_name, canonical_concept,
                    value_json, value_type, unit, evidence_refs_json,
                    confidence, status, extractor_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

    def get_assertions_for_document(self, document_id: str) -> List[MetadataAssertion]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metadata_assertions WHERE document_id = ?", (document_id,))
            rows = cursor.fetchall()
            return [self._row_to_assertion(r) for r in rows]

    def find_assertions_by_concept(self, canonical_concept: str) -> List[MetadataAssertion]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metadata_assertions WHERE canonical_concept = ?", (canonical_concept,))
            rows = cursor.fetchall()
            return [self._row_to_assertion(r) for r in rows]

    def _row_to_assertion(self, row) -> MetadataAssertion:
        return MetadataAssertion(
            assertion_id=row[0],
            document_id=row[1],
            raw_attribute_name=row[2],
            canonical_concept=row[3],
            value=json.loads(row[4]),
            value_type=row[5],
            unit=row[6],
            evidence_refs=json.loads(row[7]),
            confidence=row[8],
            status=AssertionStatus(row[9]),
            extractor_version=row[10]
        )

    def save_relationship(self, rel: EvidenceRelationship):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evidence_relationships (
                    relationship_id, source_id, target_id, relationship_type,
                    confidence, evidence_refs_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rel.relationship_id,
                rel.source_id,
                rel.target_id,
                rel.relationship_type.value if isinstance(rel.relationship_type, RelationshipType) else str(rel.relationship_type),
                rel.confidence,
                json.dumps(rel.evidence_refs),
                json.dumps(rel.metadata)
            ))
            conn.commit()

    def get_relationships_for_source(self, source_id: str) -> List[EvidenceRelationship]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evidence_relationships WHERE source_id = ?", (source_id,))
            rows = cursor.fetchall()
            return [
                EvidenceRelationship(
                    relationship_id=r[0],
                    source_id=r[1],
                    target_id=r[2],
                    relationship_type=RelationshipType(r[3]),
                    confidence=r[4],
                    evidence_refs=json.loads(r[5]),
                    metadata=json.loads(r[6])
                )
                for r in rows
            ]

assertion_store = AssertionStore()

class AssertionExtractor:
    """
    Extracts open-world metadata assertions from document segments and structured tables.
    Preserves exact provenance (chunk_id and page_number) for each asserted fact.
    """
    
    NUMERIC_LINE = re.compile(r'([A-Za-z0-9\s,\-_/]+?)\s*[:=]\s*([$€£₹]?\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:cr|crore|crores|million|billion|thousand|%)?)', re.IGNORECASE)

    def extract_assertions(self, passport: DocumentPassport, segments: List[ParsedSegment]) -> List[MetadataAssertion]:
        assertions: List[MetadataAssertion] = []
        
        # 1. Base assertions from Passport
        if passport.entity:
            assertions.append(self._make_assertion(
                passport.document_id, "entity_name", "entity", passport.entity, "string", None,
                [f"{passport.document_id}_p1"], 1.0
            ))
        if passport.fiscal_year:
            assertions.append(self._make_assertion(
                passport.document_id, "reporting_fiscal_year", "fiscal_year", passport.fiscal_year, "string", None,
                [f"{passport.document_id}_p1"], 1.0
            ))
        if passport.scope:
            assertions.append(self._make_assertion(
                passport.document_id, "reporting_scope", "scope", passport.scope, "string", None,
                [f"{passport.document_id}_p1"], 0.95
            ))

        # 2. Extract assertions from tables
        for seg in segments:
            if seg.segment_type == "table" and seg.table_headers and seg.table_rows:
                headers = seg.table_headers
                for row_idx, row in enumerate(seg.table_rows):
                    if not row or len(row) < 2:
                        continue
                    row_label = row[0]
                    if not row_label or len(row_label.strip()) < 2:
                        continue

                    # For each cell in the row
                    for col_idx, cell_val in enumerate(row[1:], start=1):
                        clean_cell = str(cell_val).strip()
                        if not clean_cell or clean_cell in ["-", "N/A", "NA", ""]:
                            continue
                        
                        col_label = headers[col_idx] if col_idx < len(headers) else f"Col_{col_idx}"
                        raw_attr = f"{row_label} - {col_label}"
                        canonical, conf = concept_registry.canonicalize(raw_attr)

                        evidence_ref = f"{passport.document_id}_p{seg.page_number}_tbl_{seg.table_id or 'table'}_r{row_idx}"
                        assertions.append(self._make_assertion(
                            passport.document_id, raw_attr, canonical, clean_cell, "string", passport.currency,
                            [evidence_ref], conf
                        ))

            elif seg.segment_type == "text":
                # Find key-value patterns in text
                lines = seg.text.split("\n")
                for line in lines[:20]: # sample top lines
                    match = self.NUMERIC_LINE.search(line)
                    if match:
                        attr_name = match.group(1).strip()
                        val_str = match.group(2).strip()
                        if len(attr_name) > 3 and not attr_name.lower().startswith("table"):
                            canonical, conf = concept_registry.canonicalize(attr_name)
                            evidence_ref = f"{passport.document_id}_p{seg.page_number}"
                            assertions.append(self._make_assertion(
                                passport.document_id, attr_name, canonical, val_str, "string", passport.currency,
                                [evidence_ref], conf
                            ))

        return assertions

    def _make_assertion(self, doc_id: str, raw_attr: str, canonical: Optional[str], val: Any, val_type: str, unit: Optional[str], evidence_refs: List[str], conf: float) -> MetadataAssertion:
        assert_id = f"ast_{doc_id}_{uuid.uuid4().hex[:8]}"
        return MetadataAssertion(
            assertion_id=assert_id,
            document_id=doc_id,
            raw_attribute_name=raw_attr,
            canonical_concept=canonical,
            value=val,
            value_type=val_type,
            unit=unit,
            evidence_refs=evidence_refs,
            confidence=conf,
            status=AssertionStatus.SUPPORTED,
            extractor_version="1.0.0"
        )

assertion_extractor = AssertionExtractor()
