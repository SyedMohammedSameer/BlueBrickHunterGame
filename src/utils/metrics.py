"""Evaluation metrics for the multi-agent system."""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for the system."""

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names to values
        """
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        }

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "binary",
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC-ROC)
            average: Averaging strategy for multi-class

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        if y_prob is not None:
            try:
                if average == "binary":
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
                else:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob, average=average, multi_class="ovr")
            except ValueError:
                # AUC-ROC cannot be calculated (e.g., only one class present)
                metrics["auc_roc"] = 0.0

        return metrics

    @staticmethod
    def retrieval_metrics(
        relevant_items: List[Any],
        retrieved_items: List[Any],
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate retrieval metrics (precision, recall, F1 @ K).

        Args:
            relevant_items: List of relevant items
            retrieved_items: List of retrieved items
            k: Number of top items to consider (if None, use all)

        Returns:
            Dictionary of metric names to values
        """
        if k is not None:
            retrieved_items = retrieved_items[:k]

        relevant_set = set(relevant_items)
        retrieved_set = set(retrieved_items)

        if len(retrieved_set) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            f"precision@{k or len(retrieved_items)}": precision,
            f"recall@{k or len(retrieved_items)}": recall,
            f"f1@{k or len(retrieved_items)}": f1,
        }

    @staticmethod
    def ndcg_score(
        relevance_scores: List[float],
        retrieved_items: List[Any],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).

        Args:
            relevance_scores: Relevance scores for retrieved items
            retrieved_items: List of retrieved items
            k: Number of top items to consider

        Returns:
            NDCG score
        """
        if k is not None:
            relevance_scores = relevance_scores[:k]

        if len(relevance_scores) == 0:
            return 0.0

        # Calculate DCG
        dcg = relevance_scores[0]
        for i, score in enumerate(relevance_scores[1:], start=2):
            dcg += score / np.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for i, score in enumerate(ideal_scores[1:], start=2):
            idcg += score / np.log2(i + 1)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ndcg

    @staticmethod
    def agent_performance_metrics(
        response_times: List[float],
        success_count: int,
        total_count: int,
    ) -> Dict[str, float]:
        """
        Calculate agent performance metrics.

        Args:
            response_times: List of response times in seconds
            success_count: Number of successful queries
            total_count: Total number of queries

        Returns:
            Dictionary of metric names to values
        """
        return {
            "success_rate": success_count / total_count if total_count > 0 else 0.0,
            "avg_response_time": np.mean(response_times) if response_times else 0.0,
            "p50_response_time": np.median(response_times) if response_times else 0.0,
            "p95_response_time": np.percentile(response_times, 95) if response_times else 0.0,
            "p99_response_time": np.percentile(response_times, 99) if response_times else 0.0,
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
        """
        Print metrics in a formatted table.

        Args:
            metrics: Dictionary of metric names to values
            title: Title for the metrics table
        """
        print(f"\n{'=' * 50}")
        print(f"{title:^50}")
        print(f"{'=' * 50}")

        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name:<30} {value:>15.4f}")
            else:
                print(f"{name:<30} {value:>15}")

        print(f"{'=' * 50}\n")
