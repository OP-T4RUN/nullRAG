from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import index_router, query_router, health_router

app = FastAPI(
    title="Smart Knowledge Base for Agents",
    description="Evidence-aware knowledge base with Document Passports, Evidence Obligation Ledgers, and Provenance Citations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, tags=["Health"])
app.include_router(index_router, tags=["Ingestion"])
app.include_router(query_router, tags=["Query"])

@app.get("/")
def root():
    return {
        "title": "Smart Knowledge Base for Agents",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
