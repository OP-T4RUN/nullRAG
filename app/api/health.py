from fastapi import APIRouter
from app.config import settings
from app.storage.vector_store import vector_storage

router = APIRouter()

@router.get("/health")
def health_check():
    collections = []
    try:
        collections = [c.name for c in vector_storage.client.get_collections().collections]
    except Exception:
        pass

    return {
        "status": "healthy",
        "app_env": settings.APP_ENV,
        "collections": collections,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_configured": bool(settings.LLM_API_KEY),
        "langfuse_configured": bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)
    }
