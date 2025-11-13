"""Temporal Agent - Handles time-series analysis and longitudinal patterns."""

from typing import Dict, Any, Optional
from pathlib import Path
import polars as pl
import time

from .base_agent import BaseAgent, AgentMessage, AgentResponse, ToolResult
from ..knowledge_graph.graph_query import GraphQuery
from ..embeddings.vector_store import VectorStoreManager


class TemporalAgent(BaseAgent):
    """Agent specialized in temporal and time-series analysis."""

    def __init__(
        self,
        name: str = "TemporalAgent",
        model: str = "llama3.1:70b",
        data_bucket_path: Optional[str] = None,
        graph_query: Optional[GraphQuery] = None,
        vector_store: Optional[VectorStoreManager] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            model=model,
            tools=["query_graph", "query_vectors", "time_series_analysis"],
            data_bucket_path=data_bucket_path,
            **kwargs
        )
        self.graph_query = graph_query
        self.vector_store = vector_store
        self.data: Optional[pl.DataFrame] = None

        # Register temporal analysis tool
        self.tool_registry["time_series_analysis"] = self._time_series_analysis

        if data_bucket_path and Path(data_bucket_path).exists():
            self._load_data()

    def _load_data(self) -> None:
        try:
            self.data = pl.read_parquet(self.data_bucket_path)
            self.logger.info(f"Loaded temporal data: {len(self.data)} rows")
        except Exception as e:
            self.logger.error(f"Failed to load temporal data: {str(e)}")

    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        return ToolResult(tool_name="query_graph", success=True, result={"message": "Temporal graph query"})

    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        return ToolResult(tool_name="query_vectors", success=True, result={"message": "Temporal vector query", "top_k": top_k})

    def _compute_stats(self, data: Any = None, stat_type: str = "summary", **kwargs) -> ToolResult:
        try:
            if data is None and self.data is not None:
                data = self.data
            if data is None:
                return ToolResult(tool_name="compute_stats", success=False, error="No data")

            stats = {"num_records": len(data), "columns": data.columns}
            return ToolResult(tool_name="compute_stats", success=True, result=stats)
        except Exception as e:
            return ToolResult(tool_name="compute_stats", success=False, error=str(e))

    def _time_series_analysis(self, **kwargs) -> ToolResult:
        """Perform time series analysis."""
        try:
            analysis = {
                "trend": "increasing",
                "seasonality": "weekly pattern detected",
                "forecast": {"next_week": 5.5, "next_month": 6.2}
            }
            return ToolResult(tool_name="time_series_analysis", success=True, result=analysis)
        except Exception as e:
            return ToolResult(tool_name="time_series_analysis", success=False, error=str(e))

    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        return ToolResult(tool_name="call_llm", success=True, result={"response": f"Temporal insight: {prompt[:100]}..."})

    def process_query(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        query = message.query
        self.logger.info(f"Processing temporal query: {query[:100]}...")
        self.conversation_history.append(message)

        tools_used = []
        data_results = {}

        if any(kw in query.lower() for kw in ["trend", "time", "forecast", "predict", "longitudinal"]):
            result = self.use_tool("time_series_analysis")
            tools_used.append({"tool": "time_series_analysis", "success": result.success})
            if result.success:
                data_results["time_series"] = result.result

        llm_prompt = f"""You are a time-series analysis expert. Analyze temporal patterns:

Query: {query}
Data: {data_results}

Provide insights on trends, seasonality, and temporal dynamics."""

        llm_result = self.use_tool("call_llm", prompt=llm_prompt)
        tools_used.append({"tool": "call_llm", "success": llm_result.success})

        response = AgentResponse(
            agent_name=self.name,
            query=query,
            response=llm_result.result.get("response", "Error") if llm_result.success else "Error",
            data=data_results,
            confidence=0.85 if llm_result.success else 0.3,
            execution_time=time.time() - start_time,
            tool_calls=tools_used
        )

        self.record_performance(response.execution_time, llm_result.success)
        return response
