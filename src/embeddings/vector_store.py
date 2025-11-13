"""Vector store management using ChromaDB."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger


class VectorStoreManager:
    """Manage vector embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: Union[str, Path],
        collection_name: str = "default"
    ):
        """
        Initialize vector store manager.

        Args:
            persist_directory: Directory for persistence
            collection_name: Collection name
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.logger = logger.bind(name=__name__)

        # Create directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self._init_chromadb()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False
            ))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "College experience embeddings"}
            )

            self.logger.info(f"ChromaDB initialized: collection={self.collection_name}")

        except ImportError:
            self.logger.warning("ChromaDB not installed. Using fallback in-memory store.")
            self._init_fallback_store()

    def _init_fallback_store(self) -> None:
        """Initialize fallback in-memory vector store."""
        self.client = None
        self.collection = None
        self.fallback_store: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to vector store.

        Args:
            ids: List of unique identifiers
            embeddings: List of embedding arrays
            documents: Optional list of source documents
            metadatas: Optional list of metadata dictionaries
        """
        # Convert numpy arrays to lists
        embedding_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

        if self.collection is not None:
            # ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embedding_lists,
                documents=documents,
                metadatas=metadatas
            )
            self.logger.info(f"Added {len(ids)} embeddings to {self.collection_name}")
        else:
            # Fallback
            for i, id_ in enumerate(ids):
                self.fallback_store[id_] = {
                    "embedding": embedding_lists[i],
                    "document": documents[i] if documents else None,
                    "metadata": metadatas[i] if metadatas else {}
                }

    def query(
        self,
        query_embeddings: Union[np.ndarray, List[np.ndarray]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query vector store for similar embeddings.

        Args:
            query_embeddings: Query embedding(s)
            n_results: Number of results to return
            where: Metadata filter
            include: Fields to include in results

        Returns:
            Query results
        """
        # Convert to list format
        if isinstance(query_embeddings, np.ndarray):
            if len(query_embeddings.shape) == 1:
                query_embeddings = [query_embeddings.tolist()]
            else:
                query_embeddings = query_embeddings.tolist()

        if self.collection is not None:
            # ChromaDB
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include or ["embeddings", "documents", "metadatas", "distances"]
            )
            return results
        else:
            # Fallback - simple cosine similarity
            return self._fallback_query(query_embeddings[0], n_results, where)

    def _fallback_query(
        self,
        query_embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback query using cosine similarity."""
        query_array = np.array(query_embedding)

        similarities = []
        for id_, data in self.fallback_store.items():
            # Check metadata filter
            if where:
                match = all(
                    data["metadata"].get(k) == v
                    for k, v in where.items()
                )
                if not match:
                    continue

            # Compute cosine similarity
            emb_array = np.array(data["embedding"])
            similarity = np.dot(query_array, emb_array) / (
                np.linalg.norm(query_array) * np.linalg.norm(emb_array) + 1e-10
            )
            similarities.append((id_, similarity, data))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Format results
        top_results = similarities[:n_results]

        return {
            "ids": [[r[0] for r in top_results]],
            "distances": [[1 - r[1] for r in top_results]],  # Convert similarity to distance
            "documents": [[r[2]["document"] for r in top_results]],
            "metadatas": [[r[2]["metadata"] for r in top_results]],
        }

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get embeddings by ID or filter.

        Args:
            ids: List of IDs to retrieve
            where: Metadata filter
            limit: Maximum number of results

        Returns:
            Retrieved embeddings
        """
        if self.collection is not None:
            return self.collection.get(
                ids=ids,
                where=where,
                limit=limit
            )
        else:
            # Fallback
            results = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

            items = []
            if ids:
                items = [(id_, self.fallback_store[id_]) for id_ in ids if id_ in self.fallback_store]
            else:
                items = list(self.fallback_store.items())

            # Apply metadata filter
            if where:
                items = [
                    (id_, data) for id_, data in items
                    if all(data["metadata"].get(k) == v for k, v in where.items())
                ]

            # Apply limit
            if limit:
                items = items[:limit]

            for id_, data in items:
                results["ids"].append(id_)
                results["documents"].append(data["document"])
                results["metadatas"].append(data["metadata"])
                results["embeddings"].append(data["embedding"])

            return results

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update embeddings in vector store.

        Args:
            ids: List of IDs to update
            embeddings: Optional new embeddings
            documents: Optional new documents
            metadatas: Optional new metadata
        """
        if embeddings:
            embeddings = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

        if self.collection is not None:
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        else:
            # Fallback
            for i, id_ in enumerate(ids):
                if id_ in self.fallback_store:
                    if embeddings:
                        self.fallback_store[id_]["embedding"] = embeddings[i]
                    if documents:
                        self.fallback_store[id_]["document"] = documents[i]
                    if metadatas:
                        self.fallback_store[id_]["metadata"] = metadatas[i]

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete embeddings from vector store.

        Args:
            ids: List of IDs to delete
            where: Metadata filter for deletion
        """
        if self.collection is not None:
            self.collection.delete(ids=ids, where=where)
        else:
            # Fallback
            if ids:
                for id_ in ids:
                    self.fallback_store.pop(id_, None)
            elif where:
                to_delete = [
                    id_ for id_, data in self.fallback_store.items()
                    if all(data["metadata"].get(k) == v for k, v in where.items())
                ]
                for id_ in to_delete:
                    del self.fallback_store[id_]

    def count(self) -> int:
        """
        Get number of embeddings in store.

        Returns:
            Number of embeddings
        """
        if self.collection is not None:
            return self.collection.count()
        else:
            return len(self.fallback_store)

    def reset(self) -> None:
        """Reset the collection."""
        if self.collection is not None:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
        else:
            self.fallback_store.clear()

        self.logger.info(f"Collection {self.collection_name} reset")
