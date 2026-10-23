import os
import json
import httpx
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings
from app.schemas.api_models import ChunkCitation
from app.schemas.internal_models import EvidenceObligation
from app.generation.numeric_verifier import NumericVerifier

class AnswerGenerator:
    """
    Synthesizes grounded answers using only the retrieved, eligible evidence.
    Enforces strict citations, addresses evidence obligations, and avoids hallucinations.
    """
    def __init__(self):
        self.verifier = NumericVerifier()

    async def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        obligations: List[EvidenceObligation],
        consistency_report: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[ChunkCitation], Dict[str, Any]]:
        # Format citations
        citations: List[ChunkCitation] = []
        context_blocks = []
        
        for c in evidence_chunks:
            citation = ChunkCitation(
                document_id=c.get("document_id", "unknown_doc"),
                page_number=c.get("page_number", 1),
                chunk_id=c.get("chunk_id", ""),
                text=c.get("text", ""),
                metadata={
                    "entity": c.get("entity"),
                    "fiscal_year": c.get("fiscal_year"),
                    "scope": c.get("scope"),
                    "segment_type": c.get("segment_type")
                }
            )
            citations.append(citation)
            
            block_header = f"[Source: {citation.document_id} | Page: {citation.page_number} | Year: {c.get('fiscal_year', 'N/A')} | Scope: {c.get('scope', 'N/A')}]"
            context_blocks.append(f"{block_header}\n{citation.text}")

        formatted_context = "\n\n---\n\n".join(context_blocks)
        
        # Build prompt
        prompt = self._build_prompt(query, formatted_context, obligations, consistency_report)
        
        # Call LLM or deterministic fallback
        if settings.LLM_API_KEY:
            try:
                answer, usage = await self._call_llm(prompt)
                from app.generation.validation import claim_validator
                val_res = claim_validator.validate_answer(answer, evidence_chunks, obligations)
                usage["validation"] = val_res
                return answer, citations, usage
            except Exception as e:
                # Fallback to structured grounded extractive synthesis
                answer = self._generate_extractive_fallback(query, evidence_chunks, obligations)
                return answer, citations, {"tokens": 0, "cost_usd": 0.0, "note": f"Fallback: {str(e)}"}
        else:
            answer = self._generate_extractive_fallback(query, evidence_chunks, obligations)
            return answer, citations, {"tokens": 0, "cost_usd": 0.0, "note": "Local extractive fallback"}

    def _build_prompt(
        self,
        query: str,
        context: str,
        obligations: List[EvidenceObligation],
        consistency_report: Optional[Dict[str, Any]] = None
    ) -> str:
        obligations_lines = []
        for o in obligations:
            status_desc = f"Status: {o.status}" if hasattr(o, "status") else f"Coverage: {o.coverage_status}"
            obligations_lines.append(f"- {o.description} ({status_desc})")
        obligations_text = "\n".join(obligations_lines)

        consistency_guidance = ""
        if consistency_report and consistency_report.get("scope_differentiations"):
            notes = [d["note"] for d in consistency_report["scope_differentiations"]]
            consistency_guidance = f"\nAccounting & Scope Distinctions to Respect:\n" + "\n".join([f"- {n}" for n in notes]) + "\n"

        return f"""You are a high-precision financial research assistant.
Answer the user's question completely and truthfully, using ONLY the verified evidence provided below.

Strict Instructions:
1. Every factual statement must be supported by the provided text. Cite the document ID and page number in brackets, e.g. [Document A, Page 12].
2. Do not mix reporting years or scopes (e.g. do not confuse FY2024 with FY2025, or consolidated figures with standalone/subsidiary figures).
3. If information requested is missing or not provided in the text, explicitly state that it is not reported. Do not invent or extrapolate figures.
4. Address each of the following required aspects:
{obligations_text}
{consistency_guidance}
Evidence Context:
{context}

Question:
{query}

Answer:"""

    async def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        # If Gemini key or provider
        if settings.LLM_PROVIDER == "gemini" or settings.LLM_API_KEY.startswith("AIzaSy"):
            return await self._call_gemini(prompt)
        else:
            return await self._call_openai_compatible(prompt)

    async def _call_gemini(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        import asyncio
        from google import genai

        api_key = settings.GEMINI_API_KEY or settings.LLM_API_KEY
        client = genai.Client(api_key=api_key)
        
        # Run sync client in async threadpool to keep FastAPI non-blocking
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.LLM_MODEL,
            contents=prompt,
            config={"temperature": 0.0}
        )

        content = response.text or ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            total_tokens = getattr(response.usage_metadata, "total_token_count", prompt_tokens + completion_tokens) or 0

        # Gemini 3.5 Flash pricing: ~$0.075 / 1M prompt tokens, ~$0.30 / 1M candidate tokens
        cost_usd = (prompt_tokens * 0.000000075) + (completion_tokens * 0.00000030)

        return content, {
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "provider": "gemini",
            "model": settings.LLM_MODEL
        }

    async def _call_openai_compatible(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise, grounded financial reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            content = data["choices"][0]["message"]["content"]
            usage_data = data.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
            
            cost_usd = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)
            
            return content, {
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "provider": "openai_compatible",
                "model": settings.LLM_MODEL
            }

    def _generate_extractive_fallback(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        obligations: List[EvidenceObligation]
    ) -> str:
        """
        High-precision deterministic grounded synthesis when LLM API key is not provided.
        Directly addresses obligations with cited evidence snippets.
        """
        if not chunks:
            return "No relevant evidence was found in the indexed collection to answer this query."

        lines = [f"Based on verified source documents:"]
        seen_pages = set()
        
        for c in chunks[:5]:
            doc_id = c.get("document_id")
            page = c.get("page_number")
            year = c.get("fiscal_year")
            key = (doc_id, page)
            if key in seen_pages:
                continue
            seen_pages.add(key)
            
            clean_text = " ".join(c.get("text", "").split()[:60])
            yr_str = f" ({year})" if year else ""
            lines.append(f"- [{doc_id}, Page {page}]{yr_str}: {clean_text}...")

        unresolved = [o.description for o in obligations if o.coverage_status == "uncovered"]
        if unresolved:
            lines.append("\nNote: The following information was not reported in the available sources:")
            for item in unresolved:
                lines.append(f"- {item}")

        return "\n".join(lines)
