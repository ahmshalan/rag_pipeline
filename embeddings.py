"""
Embeddings abstraction layer.

Supports multiple embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002)
- Sentence Transformers (local models)

Author: Ahmed Hossam
"""

from typing import List, Literal
import numpy as np
from abc import ABC, abstractmethod
import config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def encode(self, texts: List[str] | str, **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Provider-specific arguments
            
        Returns:
            numpy array of embeddings
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI embeddings provider.
    
    Supports:
    - text-embedding-3-small (1536 dims, $0.02/1M tokens) - Recommended
    - text-embedding-3-large (3072 dims, $0.13/1M tokens)
    - text-embedding-ada-002 (1536 dims, $0.10/1M tokens) - Legacy
    
    Benefits:
    - High quality embeddings
    - No local model download
    - Consistent performance
    - Regular updates
    
    Costs (approximate):
    - 10K queries: ~$0.20 (3-small) or ~$1.30 (3-large)
    - 100K queries: ~$2 (3-small) or ~$13 (3-large)
    """
    
    # Embedding dimensions for each model
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            batch_size: Number of texts to encode per API call
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        
        self._model_name = model
        self.batch_size = batch_size
        
        # Validate model
        if model not in self.MODEL_DIMS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported: {list(self.MODEL_DIMS.keys())}"
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        print(f"✓ OpenAI embeddings initialized: {model}")
    
    def encode(
        self,
        texts: List[str] | str,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts using OpenAI API.
        
        Args:
            texts: Single text or list of texts
            show_progress_bar: Show progress (for compatibility, uses print)
            convert_to_numpy: Return numpy array
            **kwargs: Ignored (for compatibility with sentence-transformers)
            
        Returns:
            numpy array of embeddings [n_texts, embedding_dim]
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        total = len(texts)
        
        if show_progress_bar and total > self.batch_size:
            print(f"Encoding {total} texts with OpenAI...")
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Call OpenAI API
            response = self.client.embeddings.create(
                input=batch,
                model=self._model_name
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            if show_progress_bar and total > self.batch_size:
                progress = min(i + self.batch_size, total)
                print(f"  Progress: {progress}/{total}")
        
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension for the model."""
        return self.MODEL_DIMS[self._model_name]
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Sentence Transformers embeddings provider (local models).
    
    Popular models:
    - all-MiniLM-L6-v2 (384 dims, 80MB) - Fast, good quality
    - all-mpnet-base-v2 (768 dims, 420MB) - Better quality, slower
    
    Benefits:
    - Free (no API costs)
    - Fast (local inference)
    - No internet required
    - Privacy (data stays local)
    
    Note: Requires model download on first use.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers.
        
        Args:
            model_name: Sentence transformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        print(f"✓ Sentence Transformer loaded: {model_name}")
    
    def encode(
        self,
        texts: List[str] | str,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts using Sentence Transformers.
        
        Args:
            texts: Single text or list of texts
            show_progress_bar: Show progress bar
            convert_to_numpy: Return numpy array
            **kwargs: Additional arguments for model.encode()
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        # Get dimension from model
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


def get_embeddings(
    provider: Literal["openai", "sentence-transformers"] = None,
    model: str = None,
    api_key: str | None = None
) -> EmbeddingProvider:
    """
    Factory function to get embedding provider.
    
    Args:
        provider: "openai" or "sentence-transformers"
        model: Model name (provider-specific)
        api_key: API key for OpenAI
        
    Returns:
        EmbeddingProvider instance
        
    Examples:
        # OpenAI (default)
        embeddings = get_embeddings()
        
        # OpenAI with specific model
        embeddings = get_embeddings(provider="openai", model="text-embedding-3-large")
        
        # Sentence Transformers
        embeddings = get_embeddings(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2"
        )
    """
    # Use config defaults if not specified
    provider = provider or config.settings.embedding_provider
    model = model or config.settings.embedding_model
    api_key = api_key or config.settings.openai_api_key
    
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=api_key)
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(model_name=model)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Use 'openai' or 'sentence-transformers'"
        )
