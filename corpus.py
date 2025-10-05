"""
Text corpus for the RAG system.

Domain: Software Engineering Best Practices

Author: Ahmed Hossam
"""

CORPUS = [
    {
        "id": "doc_001",
        "text": "Code reviews are essential for maintaining code quality. They help catch bugs early, "
                "ensure coding standards are followed, and facilitate knowledge sharing among team members. "
                "Aim for reviews within 24 hours to maintain momentum.",
        "metadata": {"category": "development", "topic": "code_review"}
    },
    {
        "id": "doc_002",
        "text": "Test-driven development (TDD) involves writing tests before implementing features. "
                "This approach leads to better-designed, more maintainable code and ensures high test coverage. "
                "Start with a failing test, write minimal code to pass it, then refactor.",
        "metadata": {"category": "development", "topic": "testing"}
    },
    {
        "id": "doc_003",
        "text": "Continuous integration (CI) automatically builds and tests code changes. "
                "This practice catches integration issues early and maintains a deployable main branch. "
                "Run tests on every commit and keep builds fast, ideally under 10 minutes.",
        "metadata": {"category": "devops", "topic": "ci_cd"}
    },
    {
        "id": "doc_004",
        "text": "Microservices architecture divides applications into small, independent services. "
                "Each service has a single responsibility and communicates via APIs. "
                "This approach improves scalability but increases operational complexity.",
        "metadata": {"category": "architecture", "topic": "microservices"}
    },
    {
        "id": "doc_005",
        "text": "API versioning is crucial for backward compatibility. Use semantic versioning (v1, v2) "
                "in your URL paths or headers. Deprecate old versions gradually with clear timelines "
                "to give clients time to migrate.",
        "metadata": {"category": "api_design", "topic": "versioning"}
    },
    {
        "id": "doc_006",
        "text": "Database indexing dramatically improves query performance. Create indexes on columns "
                "frequently used in WHERE clauses and JOIN conditions. However, indexes slow down "
                "writes, so balance read vs write performance needs.",
        "metadata": {"category": "database", "topic": "performance"}
    },
    {
        "id": "doc_007",
        "text": "Error handling should be consistent and informative. Use proper HTTP status codes, "
                "provide meaningful error messages, and log errors with context for debugging. "
                "Never expose internal stack traces to end users for security reasons.",
        "metadata": {"category": "api_design", "topic": "error_handling"}
    },
    {
        "id": "doc_008",
        "text": "Rate limiting protects APIs from abuse and ensures fair resource allocation. "
                "Implement token bucket or sliding window algorithms. Return 429 status codes "
                "with Retry-After headers when limits are exceeded.",
        "metadata": {"category": "api_design", "topic": "rate_limiting"}
    },
    {
        "id": "doc_009",
        "text": "Monitoring and observability are critical for production systems. Track key metrics "
                "like latency, error rates, and throughput. Use distributed tracing to debug "
                "issues across microservices and set up alerts for anomalies.",
        "metadata": {"category": "devops", "topic": "monitoring"}
    },
    {
        "id": "doc_010",
        "text": "Security best practices include principle of least privilege, input validation, "
                "and encryption for sensitive data. Regularly update dependencies to patch "
                "vulnerabilities and conduct security audits periodically.",
        "metadata": {"category": "security", "topic": "best_practices"}
    },
    {
        "id": "doc_011",
        "text": "Documentation is often overlooked but essential. Keep API docs updated with examples, "
                "write clear README files, and document architectural decisions. Good documentation "
                "reduces onboarding time and support burden.",
        "metadata": {"category": "development", "topic": "documentation"}
    },
    {
        "id": "doc_012",
        "text": "Caching strategies can significantly reduce load on databases and external APIs. "
                "Use Redis or Memcached for distributed caching. Consider cache invalidation "
                "strategies carefully - cache invalidation is one of the hardest problems in CS.",
        "metadata": {"category": "performance", "topic": "caching"}
    },
    {
        "id": "doc_013",
        "text": "Feature flags enable gradual rollouts and A/B testing. They allow deploying code "
                "without exposing features immediately. This reduces risk and enables quick "
                "rollbacks without redeployment if issues arise.",
        "metadata": {"category": "deployment", "topic": "feature_flags"}
    },
    {
        "id": "doc_014",
        "text": "Load balancing distributes traffic across multiple servers to improve reliability "
                "and performance. Use round-robin, least connections, or weighted algorithms "
                "based on your needs. Health checks ensure traffic only goes to healthy instances.",
        "metadata": {"category": "infrastructure", "topic": "load_balancing"}
    },
    {
        "id": "doc_015",
        "text": "Logging best practices include structured logging with JSON format, appropriate "
                "log levels (DEBUG, INFO, WARNING, ERROR), and correlation IDs for tracing "
                "requests. Avoid logging sensitive information and use log aggregation tools.",
        "metadata": {"category": "devops", "topic": "logging"}
    }
]


def get_corpus():
    """Return the text corpus."""
    return CORPUS


def get_corpus_texts():
    """Return only the text content of the corpus."""
    return [doc["text"] for doc in CORPUS]


def get_corpus_ids():
    """Return only the document IDs."""
    return [doc["id"] for doc in CORPUS]
