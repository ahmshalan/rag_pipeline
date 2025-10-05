"""
LanceDB vector store integration.

Provides persistent vector storage with efficient similarity search.

Author: Ahmed Hossam
"""

import os
from typing import List, Literal, Optional
import numpy as np
import lancedb
import config
from corpus import get_corpus
from embeddings import get_embeddings


class LanceDBVectorStore:
    """
    LanceDB-based vector store for document embeddings.
    
    Benefits over in-memory storage:
    - Persistent: Survives restarts
    - Scalable: Handles millions of vectors efficiently
    - Fast: Optimized vector search with approximate nearest neighbor (ANN)
    - Production-ready: Used by companies like Scale AI, Midjourney
    
    LanceDB is embedded (no separate server needed) making it perfect for
    this use case - production-quality without operational overhead.
    """
    
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        db_uri: str = config.LANCEDB_URI,
        table_name: str = config.LANCEDB_TABLE_NAME,
        embedding_provider: str = config.EMBEDDING_PROVIDER
    ):
        self.model_name = model_name
        self.db_uri = db_uri
        self.table_name = table_name
        self.embedding_provider = embedding_provider
        
        # Initialize embeddings
        self.embeddings = get_embeddings(
            provider=embedding_provider,
            model=model_name
        )
        
        # Connect to LanceDB
        self.db = lancedb.connect(db_uri)
        
        # Initialize or load table
        self.table = self._init_table()
        
        print(f"✓ LanceDB initialized at {db_uri}")
        print(f"✓ Table '{table_name}' contains {self.table.count_rows()} documents")
    
    def _init_table(self):
        """Initialize or load the LanceDB table."""
        # Check if table exists
        table_names = self.db.table_names()
        
        if self.table_name in table_names:
            print(f"Loading existing table '{self.table_name}'...")
            return self.db.open_table(self.table_name)
        else:
            print(f"Creating new table '{self.table_name}'...")
            return self._create_and_populate_table()
    
    def _create_and_populate_table(self):
        """Create table and populate with corpus documents."""
        corpus = get_corpus()
        
        # Generate embeddings
        texts = [doc["text"] for doc in corpus]
        print(f"Encoding {len(texts)} documents with {self.embedding_provider}...")
        embeddings_array = self.embeddings.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Prepare data for LanceDB
        data = []
        for doc, embedding in zip(corpus, embeddings_array):
            data.append({
                "doc_id": doc["id"],
                "text": doc["text"],
                "category": doc["metadata"]["category"],
                "topic": doc["metadata"]["topic"],
                "vector": embedding.tolist()
            })
        
        # Create table
        table = self.db.create_table(self.table_name, data=data)
        print(f"✓ Created table with {len(data)} documents")
        
        return table
    
    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        metric: Literal["cosine", "dot"] = "cosine"
    ) -> List[dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            metric: Distance metric ("cosine" or "dot")
            
        Returns:
            List of documents with similarity scores
        """
        # Encode query
        query_embedding = self.embeddings.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        # LanceDB uses L2 distance by default, but we can use cosine
        # by normalizing vectors (which we do for cosine similarity)
        if metric == "cosine":
            # Normalize query vector for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in LanceDB
        results = (
            self.table.search(query_embedding.tolist())
            .limit(top_k)
            .to_list()
        )
        
        # Format results
        formatted_results = []
        for result in results:
            # Calculate similarity score
            # LanceDB returns _distance (L2 distance)
            # Convert to similarity score
            if metric == "cosine":
                # For normalized vectors, L2 distance relates to cosine similarity:
                # similarity = 1 - (distance^2 / 2)
                distance = result.get("_distance", 0)
                similarity_score = 1 - (distance ** 2 / 2)
            else:
                # For dot product, we'll compute it directly
                doc_vector = np.array(result["vector"])
                similarity_score = float(np.dot(query_embedding, doc_vector))
            
            formatted_results.append({
                "doc_id": result["doc_id"],
                "text": result["text"],
                "metadata": {
                    "category": result["category"],
                    "topic": result["topic"]
                },
                "similarity_score": float(similarity_score)
            })
        
        return formatted_results
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        category: str,
        topic: str
    ) -> None:
        """
        Add a new document to the vector store.
        
        Args:
            doc_id: Unique document ID
            text: Document text
            category: Document category
            topic: Document topic
        """
        # Generate embedding
        embedding = self.embeddings.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        # Add to table
        self.table.add([{
            "doc_id": doc_id,
            "text": text,
            "category": category,
            "topic": topic,
            "vector": embedding.tolist()
        }])
        
        print(f"✓ Added document {doc_id}")
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID to delete
        """
        self.table.delete(f"doc_id = '{doc_id}'")
        print(f"✓ Deleted document {doc_id}")
    
    def reset(self) -> None:
        """Drop the table and recreate it with corpus."""
        self.db.drop_table(self.table_name)
        print(f"✓ Dropped table '{self.table_name}'")
        self.table = self._create_and_populate_table()
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "table_name": self.table_name,
            "document_count": self.table.count_rows(),
            "db_uri": self.db_uri,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.model_name,
            "embedding_dim": self.embeddings.embedding_dim
        }


class InMemoryVectorStore:
    """
    Fallback in-memory vector store (original implementation).
    
    Used when USE_LANCEDB is False or for testing.
    """
    
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        embedding_provider: str = config.EMBEDDING_PROVIDER
    ):
        self.model_name = model_name
        self.embedding_provider = embedding_provider
        
        # Initialize embeddings
        self.embeddings = get_embeddings(
            provider=embedding_provider,
            model=model_name
        )
        
        # Load and encode corpus
        self.corpus = get_corpus()
        self.corpus_texts = [doc["text"] for doc in self.corpus]
        
        print(f"Encoding {len(self.corpus_texts)} documents...")
        self.corpus_embeddings = self.embeddings.encode(
            self.corpus_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"✓ In-memory store: {len(self.corpus)} documents encoded")
    
    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        metric: Literal["cosine", "dot"] = "cosine"
    ) -> List[dict]:
        """Search using in-memory vectors."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Encode query
        query_embedding = self.embeddings.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Compute similarities
        if metric == "cosine":
            similarities = cosine_similarity(
                query_embedding,
                self.corpus_embeddings
            )[0]
        else:  # dot product
            similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            results.append({
                "doc_id": self.corpus[idx]["id"],
                "text": self.corpus[idx]["text"],
                "metadata": self.corpus[idx]["metadata"],
                "similarity_score": float(similarities[idx])
            })
        
        return results
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "storage_type": "in-memory",
            "document_count": len(self.corpus),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.model_name,
            "embedding_dim": self.embeddings.embedding_dim
        }


def get_vector_store(
    model_name: str = config.EMBEDDING_MODEL,
    use_lancedb: bool = config.USE_LANCEDB
):
    """
    Factory function to get the appropriate vector store.
    
    Args:
        model_name: Embedding model name
        use_lancedb: Whether to use LanceDB (True) or in-memory (False)
        
    Returns:
        Vector store instance (LanceDB or in-memory)
    """
    if use_lancedb:
        return LanceDBVectorStore(model_name=model_name)
    else:
        return InMemoryVectorStore(model_name=model_name)
