"""Embedding generation using Ollama."""

from typing import List, Dict, Any, Union
import numpy as np
from loguru import logger
import requests
import json


class EmbeddingGenerator:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768
    ):
        """
        Initialize embedding generator.

        Args:
            base_url: Ollama API base URL
            model: Embedding model name
            dimension: Embedding dimension
        """
        self.base_url = base_url
        self.model = model
        self.dimension = dimension
        self.logger = logger.bind(name=__name__)

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to Ollama API."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                self.logger.info(f"Connected to Ollama at {self.base_url}")
            else:
                self.logger.warning(f"Ollama connection test returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama: {str(e)}")
            self.logger.info("Embeddings will be generated when Ollama is available")

    def generate(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text.

        Args:
            text: Text string or list of strings
            normalize: Whether to normalize embeddings

        Returns:
            Embedding array or list of embedding arrays
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings = []

        for txt in texts:
            try:
                embedding = self._generate_single(txt)
                if normalize:
                    embedding = self._normalize(embedding)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Failed to generate embedding: {str(e)}")
                # Return zero vector as fallback
                embeddings.append(np.zeros(self.dimension))

        return embeddings[0] if is_single else embeddings

    def _generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding array
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        embedding = np.array(result["embedding"])

        return embedding

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length.

        Args:
            embedding: Input embedding

        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress

        Returns:
            List of embeddings
        """
        embeddings = []

        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if show_progress:
                batch_num = i // batch_size + 1
                self.logger.info(f"Processing batch {batch_num}/{num_batches}")

            batch_embeddings = self.generate(batch, normalize=normalize)
            embeddings.extend(batch_embeddings)

        return embeddings

    def create_student_narrative(
        self,
        student_id: str,
        behavior_summary: Dict[str, Any]
    ) -> str:
        """
        Create narrative description from student behavior data.

        Args:
            student_id: Student identifier
            behavior_summary: Dictionary with behavior data

        Returns:
            Narrative text
        """
        narrative_parts = [f"Student {student_id}"]

        # Add location information
        if "locations_visited" in behavior_summary:
            locations = behavior_summary["locations_visited"]
            if locations:
                location_labels = [loc.get("semantic_label", "unknown") for loc in locations[:5]]
                narrative_parts.append(f"frequently visits {', '.join(location_labels)}")

        # Add activity information
        if "activities_performed" in behavior_summary:
            activities = behavior_summary["activities_performed"]
            if activities:
                activity_types = [act.get("activity_type", "unknown") for act in activities[:5]]
                narrative_parts.append(f"performs activities including {', '.join(activity_types)}")

        # Add mental health information
        if "mental_health_states" in behavior_summary:
            mh_states = behavior_summary["mental_health_states"]
            if mh_states:
                phq4_scores = [s.get("phq4_score") for s in mh_states if s.get("phq4_score") is not None]
                if phq4_scores:
                    avg_score = np.mean(phq4_scores)
                    narrative_parts.append(f"has average PHQ4 score of {avg_score:.1f}")

        return ". ".join(narrative_parts) + "."

    def create_context_embedding(
        self,
        context_type: str,
        context_data: Dict[str, Any]
    ) -> str:
        """
        Create context description for embedding.

        Args:
            context_type: Type of context (location, activity, mental_health)
            context_data: Context data

        Returns:
            Context description
        """
        if context_type == "location":
            return f"Location: {context_data.get('semantic_label', 'unknown')} with {context_data.get('visit_count', 0)} visits"
        elif context_type == "activity":
            return f"Activity: {context_data.get('activity_type', 'unknown')} with duration {context_data.get('duration', 0)}"
        elif context_type == "mental_health":
            return f"Mental health state: PHQ4 score {context_data.get('phq4_score', 'N/A')}, anxiety {context_data.get('anxiety_score', 'N/A')}, depression {context_data.get('depression_score', 'N/A')}"
        else:
            return str(context_data)
