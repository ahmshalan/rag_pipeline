"""
Retrieval engine with configurable similarity metrics.

Supports two index configurations:
1. Cosine similarity (normalized dot product)
2. Dot product similarity (raw dot product)

Now powered by LanceDB for persistent vector storage!

Author: Ahmed Hossam
"""

from typing import List, Literal
import config
from vector_store import get_vector_store


class RetrievalEngine:
    """
    Retrieval engine supporting multiple similarity metrics.
    
    Now uses LanceDB for persistent vector storage instead of in-memory.
    
    Comparison of index configs:
    
    1. COSINE SIMILARITY (DEFAULT CHOICE):
       - Normalizes vectors to unit length, measures angle between vectors
       - Range: [-1, 1] where 1 is most similar
       - Pros: Scale-invariant, works well for text of varying lengths
       - Cons: Slightly more computation due to normalization
       - Best for: General text retrieval where document length varies
    
    2. DOT PRODUCT:
       - Raw dot product of vectors without normalization
       - Range: Unbounded, depends on vector magnitudes
       - Pros: Faster computation, no normalization overhead
       - Cons: Biased toward longer documents (higher magnitude vectors)
       - Best for: When all documents are similar length or magnitude matters
       - IMPORTANT: Use models trained for dot product similarity (e.g., 
         sentence-transformers models with 'dot' in the name, or models 
         specifically documented as optimized for dot product). Many models 
         are trained with cosine similarity and may not work well with dot product.
    
    JUSTIFIED CHOICE: COSINE SIMILARITY
    Reason: More robust for text retrieval where document lengths vary.
    The normalization ensures semantic similarity isn't confused with
    document length, leading to more relevant results.
    """
    
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        similarity_metric: Literal["cosine", "dot"] = config.SIMILARITY_METRIC,
        use_lancedb: bool = config.USE_LANCEDB
    ):
        self.model_name = model_name
        self.similarity_metric = similarity_metric
        self.use_lancedb = use_lancedb
        
        # Initialize vector store (LanceDB or in-memory)
        self.vector_store = get_vector_store(
            model_name=model_name,
            use_lancedb=use_lancedb
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K
    ) -> List[dict]:
        """
        Retrieve top-k most similar documents for a query.
        
        Uses LanceDB for efficient vector similarity search.
        
        Args:
            query: User's query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of dicts containing document info and similarity scores
        """
        # Delegate to vector store
        return self.vector_store.search(
            query=query,
            top_k=top_k,
            metric=self.similarity_metric
        )
    
    def generate_naive_answer(
        self,
        query: str,
        retrieved_docs: List[dict]
    ) -> str:
        """
        Generate a naive answer from retrieved documents.
        
        This is intentionally simple - just concatenates top snippets.
        In production, you'd use an LLM here.
        
        Args:
            query: User's query
            retrieved_docs: Retrieved documents from retrieve()
            
        Returns:
            Naive answer string
        """
        if not retrieved_docs:
            return "No relevant information found."
        
        # Simple approach: return top document's text with light formatting
        top_doc = retrieved_docs[0]
        
        answer_parts = [
            f"Based on the retrieved information:",
            f"\n{top_doc['text']}",
        ]
        
        # Add reference to other relevant docs if available
        if len(retrieved_docs) > 1:
            other_topics = [
                doc['metadata'].get('topic', 'related')
                for doc in retrieved_docs[1:]
            ]
            answer_parts.append(
                f"\n\nRelated topics: {', '.join(other_topics)}"
            )
        
        return "\n".join(answer_parts)


    def get_stats(self) -> dict:
        """Get vector store statistics."""
        stats = self.vector_store.get_stats()
        stats["similarity_metric"] = self.similarity_metric
        return stats

