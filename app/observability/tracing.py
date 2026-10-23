import time
from typing import List, Dict, Any, Optional
from app.config import settings
from app.schemas.api_models import TraceStep

try:
    from langfuse import Langfuse
    has_langfuse = True
except ImportError:
    has_langfuse = False

class ExecutionTracer:
    """
    Manages both:
    1. The required JSON array response trace for /query endpoint
    2. Langfuse telemetry/tracing for latency and cost tracking
    """
    def __init__(self, trace_name: str = "query_execution", collection_name: str = "default", query: str = ""):
        self.trace_name = trace_name
        self.collection_name = collection_name
        self.query = query
        self.steps: List[TraceStep] = []
        self.start_time = time.time()
        self.langfuse_client = None
        self.langfuse_trace = None
        
        # Initialize Langfuse if credentials are provided
        if has_langfuse and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                self.langfuse_client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                self.langfuse_trace = self.langfuse_client.trace(
                    name=trace_name,
                    metadata={"collection_name": collection_name},
                    input={"query": query}
                )
            except Exception as e:
                # Fallback to local logging if Langfuse server is unreachable
                self.langfuse_client = None
                self.langfuse_trace = None

    def add_step(self, step_name: str, description: str, step_input: Any, step_output: Any, duration_ms: Optional[float] = None) -> TraceStep:
        step = TraceStep(
            step_name=step_name,
            description=description,
            input=step_input,
            output=step_output
        )
        self.steps.append(step)
        
        # Log span to Langfuse if enabled
        if self.langfuse_trace:
            try:
                self.langfuse_trace.span(
                    name=step_name,
                    input=step_input,
                    output=step_output,
                    metadata={"description": description, "duration_ms": duration_ms}
                )
            except Exception:
                pass
                
        return step

    def finalize(self, answer: str, total_tokens: Optional[int] = None, cost_usd: Optional[float] = None) -> List[TraceStep]:
        total_duration = time.time() - self.start_time
        if self.langfuse_trace:
            try:
                self.langfuse_trace.update(
                    output={"answer": answer},
                    metadata={
                        "total_duration_sec": total_duration,
                        "total_steps": len(self.steps),
                        "total_tokens": total_tokens,
                        "cost_usd": cost_usd
                    }
                )
                if self.langfuse_client:
                    self.langfuse_client.flush()
            except Exception:
                pass
        return self.steps
