"""
Configuration settings for the RAG pipeline.

Uses Pydantic BaseSettings to load configuration from environment variables
and .env files. This allows for flexible deployment across different environments.

Author: Ahmed Hossam
"""

from typing import Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    Environment variables take precedence over .env file values.
    Variable names are case-insensitive.
    
    Example .env file:
        EMBEDDING_MODEL=all-MiniLM-L6-v2
        DEFAULT_TOP_K=3
        OPENAI_API_KEY=sk-...
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Retrieval settings
    embedding_provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai",
        description="Embedding provider: openai or sentence-transformers"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (provider-specific)"
    )
    default_top_k: int = Field(
        default=3,
        description="Default number of documents to retrieve",
        ge=1
    )
    max_top_k: int = Field(
        default=10,
        description="Maximum allowed top_k value",
        ge=1
    )
    
    # LanceDB settings
    lancedb_uri: str = Field(
        default="./data/lancedb",
        description="Path to LanceDB storage"
    )
    lancedb_table_name: str = Field(
        default="documents",
        description="Name of the LanceDB table"
    )
    use_lancedb: bool = Field(
        default=True,
        description="Use LanceDB for vector storage (False for in-memory)"
    )
    
    # Guardrail settings - Basic
    denylist: List[str] = Field(
        default=[
            "password",
            "secret",
            "api_key",
            "token",
            "credit_card",
            "ssn",
            "social security",
        ],
        description="List of forbidden terms in queries"
    )
    max_query_length: int = Field(
        default=500,
        description="Maximum query length in characters",
        ge=1
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit for API requests",
        ge=1
    )
    
    # Semantic Guardrails settings
    semantic_guardrail_enabled: bool = Field(
        default=True,
        description="Enable semantic guardrails"
    )
    semantic_guardrail_threshold: float = Field(
        default=0.75,
        description="Similarity threshold for blocking dangerous queries",
        ge=0.0,
        le=1.0
    )
    semantic_guardrail_model: str = Field(
        default="text-embedding-3-small",
        description="Model for semantic guardrails"
    )
    semantic_guardrail_provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai",
        description="Provider for semantic guardrails embeddings"
    )
    
    # LLM Guardrails settings
    llm_guardrail_enabled: bool = Field(
        default=False,
        description="Enable LLM-based guardrails (requires API key)"
    )
    llm_guardrail_provider: Literal["openai", "local"] = Field(
        default="openai",
        description="LLM provider for guardrails"
    )
    llm_guardrail_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for guardrails"
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for LLM guardrails"
    )
    
    # Index configuration options
    similarity_metric: Literal["cosine", "dot"] = Field(
        default="cosine",
        description="Similarity metric for vector search"
    )
    
    # Monitoring settings
    metrics_window_size: int = Field(
        default=1000,
        description="Number of recent requests to keep for metrics",
        ge=1
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port",
        ge=1,
        le=65535
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )


# Global settings instance
settings = Settings()



# These allow existing code to import the attributes
EMBEDDING_PROVIDER = settings.embedding_provider
EMBEDDING_MODEL = settings.embedding_model
DEFAULT_TOP_K = settings.default_top_k
MAX_TOP_K = settings.max_top_k
LANCEDB_URI = settings.lancedb_uri
LANCEDB_TABLE_NAME = settings.lancedb_table_name
USE_LANCEDB = settings.use_lancedb
DENYLIST = settings.denylist
MAX_QUERY_LENGTH = settings.max_query_length
RATE_LIMIT_PER_MINUTE = settings.rate_limit_per_minute
SEMANTIC_GUARDRAIL_ENABLED = settings.semantic_guardrail_enabled
SEMANTIC_GUARDRAIL_THRESHOLD = settings.semantic_guardrail_threshold
SEMANTIC_GUARDRAIL_MODEL = settings.semantic_guardrail_model
SEMANTIC_GUARDRAIL_PROVIDER = settings.semantic_guardrail_provider
LLM_GUARDRAIL_ENABLED = settings.llm_guardrail_enabled
LLM_GUARDRAIL_PROVIDER = settings.llm_guardrail_provider
LLM_GUARDRAIL_MODEL = settings.llm_guardrail_model
OPENAI_API_KEY = settings.openai_api_key
SIMILARITY_METRIC = settings.similarity_metric
METRICS_WINDOW_SIZE = settings.metrics_window_size