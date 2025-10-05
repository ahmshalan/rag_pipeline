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


def compare_configs(query: str = "How to improve code quality?"):
    """
    Compare cosine vs dot product retrieval for analysis.
    
    Usage: python -c "from retrieval import compare_configs; compare_configs()"
    """
    print("=== Comparing Retrieval Configurations ===\n")
    print(f"Query: {query}\n")
    
    # Test with k=3
    print("--- Configuration 1: Cosine Similarity, k=3 (LanceDB) ---")
    engine_cosine = RetrievalEngine(similarity_metric="cosine")
    results_cosine = engine_cosine.retrieve(query, top_k=3)
    for i, doc in enumerate(results_cosine, 1):
        print(f"{i}. [{doc['doc_id']}] Score: {doc['similarity_score']:.4f}")
        print(f"   {doc['text'][:100]}...\n")
    
    print("\n--- Configuration 2: Dot Product, k=3 (LanceDB) ---")
    engine_dot = RetrievalEngine(similarity_metric="dot")
    results_dot = engine_dot.retrieve(query, top_k=3)
    for i, doc in enumerate(results_dot, 1):
        print(f"{i}. [{doc['doc_id']}] Score: {doc['similarity_score']:.4f}")
        print(f"   {doc['text'][:100]}...\n")
    
    # Test with k=5
    print("\n--- Configuration 3: Cosine Similarity, k=5 (LanceDB) ---")
    results_k5 = engine_cosine.retrieve(query, top_k=5)
    for i, doc in enumerate(results_k5, 1):
        print(f"{i}. [{doc['doc_id']}] Score: {doc['similarity_score']:.4f}")
    
    print("\n=== Analysis ===")
    print(f"Cosine scores are normalized: {all(abs(d['similarity_score']) <= 1 for d in results_cosine)}")
    print(f"Dot product scores are unbounded: {any(abs(d['similarity_score']) > 1 for d in results_dot)}")
    print("\nRecommendation: Use cosine similarity for production (better normalization)")
    print(f"\nVector Store Stats: {engine_cosine.get_stats()}")


if __name__ == "__main__":
    compare_configs()
