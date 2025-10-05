"""
Lightweight monitoring for the RAG system.

Tracks two key metrics:
1. Latency (p50, p95, p99) - Response time percentiles
2. Cache Hit Rate / Retrieval Drift - Tracks query diversity and result overlap

These metrics help identify:
- Performance degradation (latency)
- Query pattern changes or corpus staleness (drift)

Author: Ahmed Hossam
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np
import config


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: datetime
    query: str
    latency_ms: float
    top_doc_id: Optional[str]
    top_similarity: float
    guardrail_violation: bool = False
    error: bool = False


class MetricsTracker:
    """
    Lightweight metrics tracker using in-memory storage.
    
    METRIC 1: LATENCY TRACKING
    - Tracks p50, p95, p99 latency percentiles
    - Why: Critical for SLA monitoring and detecting performance degradation
    - How: Rolling window of recent request latencies
    
    METRIC 2: RETRIEVAL DRIFT
    - Tracks diversity of top retrieved documents
    - Measures if same docs are repeatedly retrieved (low drift = possible staleness)
    - Why: Helps detect if corpus needs updating or if queries are too homogeneous
    - How: Sliding window of top doc IDs, calculate uniqueness ratio
    
    In production, you'd export these to Prometheus, DataDog, or CloudWatch.
    """
    
    def __init__(self, window_size: int = config.METRICS_WINDOW_SIZE):
        self.window_size = window_size
        self.requests: deque[RequestMetrics] = deque(maxlen=window_size)
        self.total_requests = 0
        self.guardrail_violations = 0
        self.errors = 0
    
    def record(
        self,
        query: str,
        latency_ms: float,
        top_doc_id: Optional[str] = None,
        top_similarity: float = 0.0,
        guardrail_violation: bool = False,
        error: bool = False
    ):
        """Record metrics for a single request."""
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            query=query,
            latency_ms=latency_ms,
            top_doc_id=top_doc_id,
            top_similarity=top_similarity,
            guardrail_violation=guardrail_violation,
            error=error
        )
        
        self.requests.append(metrics)
        self.total_requests += 1
        
        if guardrail_violation:
            self.guardrail_violations += 1
        if error:
            self.errors += 1
    
    def get_latency_percentiles(self) -> dict:
        """
        Calculate latency percentiles.
        
        Returns:
            Dict with p50, p95, p99 latency in milliseconds
        """
        if not self.requests:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        
        latencies = [r.latency_ms for r in self.requests if not r.error]
        
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "count": len(latencies)
        }
    
    def get_retrieval_drift_metrics(self) -> dict:
        """
        Calculate retrieval drift/diversity metrics.
        
        Higher uniqueness ratio = more diverse queries/results (good)
        Lower uniqueness ratio = same docs repeatedly retrieved (potential issue)
        
        Returns:
            Dict with uniqueness ratio and top document frequencies
        """
        if not self.requests:
            return {
                "uniqueness_ratio": 0.0,
                "unique_docs": 0,
                "total_retrievals": 0,
                "top_3_docs": []
            }
        
        # Get all top doc IDs (excluding errors and guardrail violations)
        top_docs = [
            r.top_doc_id
            for r in self.requests
            if r.top_doc_id and not r.error and not r.guardrail_violation
        ]
        
        if not top_docs:
            return {
                "uniqueness_ratio": 0.0,
                "unique_docs": 0,
                "total_retrievals": 0,
                "top_3_docs": []
            }
        
        unique_docs = len(set(top_docs))
        total_retrievals = len(top_docs)
        uniqueness_ratio = unique_docs / total_retrievals
        
        # Get top 3 most frequently retrieved docs
        doc_counts = {}
        for doc_id in top_docs:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        top_3 = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "uniqueness_ratio": round(uniqueness_ratio, 3),
            "unique_docs": unique_docs,
            "total_retrievals": total_retrievals,
            "top_3_docs": [{"doc_id": doc_id, "count": count} for doc_id, count in top_3]
        }
    
    def get_summary(self) -> dict:
        """
        Get comprehensive metrics summary.
        
        This is what you'd expose via a /metrics endpoint or push to monitoring.
        """
        latency = self.get_latency_percentiles()
        drift = self.get_retrieval_drift_metrics()
        
        success_rate = 0.0
        if self.total_requests > 0:
            success_count = self.total_requests - self.errors - self.guardrail_violations
            success_rate = success_count / self.total_requests
        
        return {
            "total_requests": self.total_requests,
            "success_rate": round(success_rate, 3),
            "guardrail_violations": self.guardrail_violations,
            "errors": self.errors,
            "latency": latency,
            "retrieval_drift": drift,
            "window_size": len(self.requests)
        }


# Global metrics tracker instance
metrics_tracker = MetricsTracker()


def get_metrics_tracker() -> MetricsTracker:
    """Get the global metrics tracker instance."""
    return metrics_tracker
