"""
FastAPI RAG service with guardrails and monitoring.

A minimal retrieval-augmented answering service with production considerations.

Author: Ahmed Hossam
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import time

import config
from retrieval import RetrievalEngine
from guardrails import get_guardrails, GuardrailViolation
from monitoring import get_metrics_tracker

# Initialize app
app = FastAPI(
    title="RAG Pipeline API",
    description="Minimal retrieval-augmented answering service with advanced guardrails",
    version="2.0.0"
)

# Initialize components
print("Initializing RAG pipeline...")
retrieval_engine = RetrievalEngine(
    similarity_metric=config.SIMILARITY_METRIC
)

# Initialize guardrails (composite by default - includes basic + semantic)
# Set LLM_GUARDRAIL_ENABLED=True and OPENAI_API_KEY env var for LLM guardrails
guardrails = get_guardrails(
    mode="composite",
    use_basic=True,
    use_semantic=config.SEMANTIC_GUARDRAIL_ENABLED,
    use_llm=config.LLM_GUARDRAIL_ENABLED
)

metrics_tracker = get_metrics_tracker()
print("RAG pipeline ready!")


# Request/Response models
class AnswerRequest(BaseModel):
    """Request model for /answer endpoint."""
    query: str = Field(..., description="User's question", min_length=1)
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        description="Number of documents to retrieve",
        ge=1,
        le=config.MAX_TOP_K
    )
    similarity_metric: Optional[Literal["cosine", "dot"]] = Field(
        default=None,
        description="Override similarity metric for this request"
    )


class RetrievedDocument(BaseModel):
    """A single retrieved document."""
    doc_id: str
    text: str
    metadata: dict
    similarity_score: float


class AnswerResponse(BaseModel):
    """Response model for /answer endpoint."""
    query: str
    answer: str
    retrieved_docs: List[RetrievedDocument]
    metadata: dict = Field(
        description="Request metadata (latency, config used, etc.)"
    )


class MetricsResponse(BaseModel):
    """Response model for /metrics endpoint."""
    total_requests: int
    success_rate: float
    guardrail_violations: int
    errors: int
    latency: dict
    retrieval_drift: dict
    window_size: int


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    version: str
    components: dict


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "POST /answer": "Submit a query and get an answer with retrieved context",
            "GET /metrics": "View monitoring metrics",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    vector_store_stats = retrieval_engine.get_stats()
    
    # Determine guardrail type
    guardrail_type = type(guardrails).__name__
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "retrieval_engine": "operational",
            "guardrails": f"operational ({guardrail_type})",
            "metrics_tracker": "operational",
            "vector_store": vector_store_stats
        }
    }


@app.post("/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_query(request: AnswerRequest):
    """
    Answer a query using retrieval-augmented generation.
    
    Process:
    1. Validate query against guardrails
    2. Retrieve top-k relevant documents
    3. Generate naive answer from retrieved context
    4. Record metrics
    
    Raises:
        400: Guardrail violation (denylist or budget rule)
        500: Internal server error
    """
    start_time = time.time()
    top_doc_id = None
    top_similarity = 0.0
    
    try:
        # Step 1: Guardrail checks
        try:
            guardrails.validate_or_raise(request.query)
        except GuardrailViolation as e:
            # Record guardrail violation in metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics_tracker.record(
                query=request.query,
                latency_ms=latency_ms,
                guardrail_violation=True
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Guardrail violation: {str(e)}"
            )
        
        # Step 2: Override similarity metric if requested
        engine = retrieval_engine
        if request.similarity_metric and request.similarity_metric != config.SIMILARITY_METRIC:
            engine = RetrievalEngine(similarity_metric=request.similarity_metric)
        
        # Step 3: Retrieve documents
        retrieved_docs = engine.retrieve(
            query=request.query,
            top_k=request.top_k
        )
        
        # Step 4: Generate naive answer
        answer = engine.generate_naive_answer(
            query=request.query,
            retrieved_docs=retrieved_docs
        )
        
        # Extract metrics
        if retrieved_docs:
            top_doc_id = retrieved_docs[0]["doc_id"]
            top_similarity = retrieved_docs[0]["similarity_score"]
        
        # Step 5: Calculate latency and record metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics_tracker.record(
            query=request.query,
            latency_ms=latency_ms,
            top_doc_id=top_doc_id,
            top_similarity=top_similarity
        )
        
        # Step 6: Build response
        return AnswerResponse(
            query=request.query,
            answer=answer,
            retrieved_docs=[
                RetrievedDocument(**doc) for doc in retrieved_docs
            ],
            metadata={
                "latency_ms": round(latency_ms, 2),
                "top_k": request.top_k,
                "similarity_metric": request.similarity_metric or config.SIMILARITY_METRIC,
                "model": config.EMBEDDING_MODEL
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (guardrail violations)
        raise
    
    except Exception as e:
        # Record error in metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics_tracker.record(
            query=request.query,
            latency_ms=latency_ms,
            error=True
        )
        
        # Return 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get monitoring metrics.
    
    Returns:
        - Latency percentiles (p50, p95, p99)
        - Retrieval drift/diversity metrics
        - Success rate
        - Error counts
    """
    summary = metrics_tracker.get_summary()
    return MetricsResponse(**summary)


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
