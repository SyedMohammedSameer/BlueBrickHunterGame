"""Mental Health Agent - Handles PHQ4 scores and mental health analysis."""

from typing import Dict, Any, Optional
from pathlib import Path
import polars as pl
import time

from .base_agent import BaseAgent, AgentMessage, AgentResponse, ToolResult
from ..knowledge_graph.graph_query import GraphQuery
from ..knowledge_graph.schema import NodeType, RelationshipType
from ..embeddings.vector_store import VectorStoreManager


class MentalHealthAgent(BaseAgent):
    """Agent specialized in mental health data analysis."""

    def __init__(
        self,
        name: str = "MentalHealthAgent",
        model: str = "meditron:70b",  # Health-specific model
        data_bucket_path: Optional[str] = None,
        graph_query: Optional[GraphQuery] = None,
        vector_store: Optional[VectorStoreManager] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            model=model,
            tools=["query_graph", "query_vectors", "compute_stats", "predict"],
            data_bucket_path=data_bucket_path,
            **kwargs
        )
        self.graph_query = graph_query
        self.vector_store = vector_store
        self.data: Optional[pl.DataFrame] = None

        # Register predict tool
        self.tool_registry["predict"] = self._predict

        if data_bucket_path and Path(data_bucket_path).exists():
            self._load_data()

    def _load_data(self) -> None:
        try:
            self.data = pl.read_parquet(self.data_bucket_path)
            self.logger.info(f"Loaded mental health data: {len(self.data)} rows")
        except Exception as e:
            self.logger.error(f"Failed to load mental health data: {str(e)}")

    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        if not self.graph_query:
            return ToolResult(tool_name="query_graph", success=False, error="Graph not available")

        try:
            if "mental health" in query.lower() or "phq" in query.lower():
                states = self.graph_query.get_nodes_by_type(NodeType.MENTAL_HEALTH_STATE)
                return ToolResult(tool_name="query_graph", success=True, result=states)

            if "student" in query.lower():
                student_id = kwargs.get("student_id")
                if student_id:
                    states = self.graph_query.get_relationships(
                        source_id=f"student_{student_id}",
                        rel_type=RelationshipType.EXPERIENCED
                    )
                    return ToolResult(tool_name="query_graph", success=True, result=states)

            return ToolResult(tool_name="query_graph", success=True, result={"message": "Query processed"})
        except Exception as e:
            return ToolResult(tool_name="query_graph", success=False, error=str(e))

    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        return ToolResult(tool_name="query_vectors", success=True, result={"message": "Vector query", "top_k": top_k})

    def _compute_stats(self, data: Any = None, stat_type: str = "summary", **kwargs) -> ToolResult:
        try:
            if data is None and self.data is not None:
                data = self.data

            if data is None:
                return ToolResult(tool_name="compute_stats", success=False, error="No data")

            stats = {}
            if stat_type == "summary":
                stats = {"num_records": len(data), "columns": data.columns}
            elif stat_type == "phq4_stats":
                phq_cols = [col for col in data.columns if "phq" in col.lower()]
                if phq_cols:
                    col = phq_cols[0]
                    stats["mean_phq4"] = data[col].mean()
                    stats["median_phq4"] = data[col].median()
                    stats["std_phq4"] = data[col].std()

            return ToolResult(tool_name="compute_stats", success=True, result=stats)
        except Exception as e:
            return ToolResult(tool_name="compute_stats", success=False, error=str(e))

    def _predict(self, **kwargs) -> ToolResult:
        """Predict future mental health scores."""
        try:
            # Placeholder for actual prediction model
            prediction = {
                "predicted_phq4": 5.2,
                "confidence_interval": [4.5, 5.9],
                "risk_level": "moderate"
            }
            return ToolResult(tool_name="predict", success=True, result=prediction)
        except Exception as e:
            return ToolResult(tool_name="predict", success=False, error=str(e))

    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        return ToolResult(tool_name="call_llm", success=True, result={"response": f"Mental health insight: {prompt[:100]}..."})

    def process_query(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        query = message.query
        self.logger.info(f"Processing mental health query: {query[:100]}...")
        self.conversation_history.append(message)

        cache_key = f"mental_health_{hash(query)}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached

        tools_used = []
        data_results = {}

        if any(kw in query.lower() for kw in ["phq", "mental health", "anxiety", "depression"]):
            result = self.use_tool("query_graph", query=query)
            tools_used.append({"tool": "query_graph", "success": result.success})
            if result.success:
                data_results["mental_health_states"] = result.result

        if any(kw in query.lower() for kw in ["predict", "forecast", "future"]):
            result = self.use_tool("predict")
            tools_used.append({"tool": "predict", "success": result.success})
            if result.success:
                data_results["prediction"] = result.result

        if any(kw in query.lower() for kw in ["average", "mean", "statistics"]):
            result = self.use_tool("compute_stats", stat_type="phq4_stats")
            tools_used.append({"tool": "compute_stats", "success": result.success})
            if result.success:
                data_results["statistics"] = result.result

        llm_prompt = f"""You are a mental health analysis expert. Based on evidence-based practices:

Query: {query}
Data: {data_results}

Provide clinical insights on mental health patterns, risk assessment, and trends. Be cautious and evidence-based."""

        llm_result = self.use_tool("call_llm", prompt=llm_prompt)
        tools_used.append({"tool": "call_llm", "success": llm_result.success})

        response = AgentResponse(
            agent_name=self.name,
            query=query,
            response=llm_result.result.get("response", "Error") if llm_result.success else "Error",
            data=data_results,
            confidence=0.90 if llm_result.success else 0.3,
            execution_time=time.time() - start_time,
            tool_calls=tools_used
        )

        self.update_cache(cache_key, response, ttl=3600)
        self.record_performance(response.execution_time, llm_result.success)
        return response
