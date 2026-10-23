from typing import List, Dict, Any, Optional
from app.knowledge.models import EvidenceRelationship, RelationshipType, MetadataAssertion
from app.knowledge.assertions import assertion_store

class ProvenanceManager:
    """
    Manages provenance links between chunks, table cells, assertions, and document versions.
    Enforces evidence integrity and connects conflicting or complementary reports.
    """
    def link_document_family(self, doc_id_a: str, doc_id_b: str, relation_type: RelationshipType, metadata: Dict[str, Any]):
        rel_id = f"rel_{doc_id_a}_{doc_id_b}_{relation_type.value}"
        rel = EvidenceRelationship(
            relationship_id=rel_id,
            source_id=doc_id_a,
            target_id=doc_id_b,
            relationship_type=relation_type,
            confidence=1.0,
            evidence_refs=[doc_id_a, doc_id_b],
            metadata=metadata
        )
        assertion_store.save_relationship(rel)

    def detect_conflicts(self, assertions: List[MetadataAssertion]) -> List[EvidenceRelationship]:
        """
        Detects potential contradictions:
        If two assertions share the same canonical_concept but carry conflicting values.
        """
        conflicts: List[EvidenceRelationship] = []
        by_concept: Dict[str, List[MetadataAssertion]] = {}
        
        for ast in assertions:
            if ast.canonical_concept and ast.canonical_concept != "unclassified_attribute":
                by_concept.setdefault(ast.canonical_concept, []).append(ast)

        for concept, group in by_concept.items():
            if len(group) >= 2:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a1, a2 = group[i], group[j]
                        if a1.document_id != a2.document_id and a1.value != a2.value:
                            # Flag potential conflict
                            rel = EvidenceRelationship(
                                relationship_id=f"conf_{a1.assertion_id}_{a2.assertion_id}",
                                source_id=a1.assertion_id,
                                target_id=a2.assertion_id,
                                relationship_type=RelationshipType.CONTRADICTS,
                                confidence=0.85,
                                evidence_refs=a1.evidence_refs + a2.evidence_refs,
                                metadata={
                                    "concept": concept,
                                    "val1": a1.value,
                                    "val2": a2.value,
                                    "doc1": a1.document_id,
                                    "doc2": a2.document_id
                                }
                            )
                            conflicts.append(rel)
                            assertion_store.save_relationship(rel)

        return conflicts

provenance_manager = ProvenanceManager()
