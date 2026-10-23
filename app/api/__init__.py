from .index import router as index_router
from .query import router as query_router
from .health import router as health_router

__all__ = ["index_router", "query_router", "health_router"]
