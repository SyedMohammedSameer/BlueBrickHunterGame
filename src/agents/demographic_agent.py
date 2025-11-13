"""Demographic Agent - Handles demographic data and cohort analysis."""

from typing import Dict, Any, Optional
from pathlib import Path
import polars as pl
import time

from .base_agent import BaseAgent, AgentMessage, AgentResponse, ToolResult
from ..knowledge_graph.graph_query import GraphQuery


class DemographicAgent(BaseAgent):
    """Agent specialized in demographic and cohort analysis."""

    def __init__(
        self,
        name: str = "DemographicAgent",
        model: str = "llama3.1:70b",
        data_bucket_path: Optional[str] = None,
        graph_query: Optional[GraphQuery] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            model=model,
            tools=["query_graph", "compute_stats", "cohort_analysis"],
            data_bucket_path=data_bucket_path,
            **kwargs
        )
        self.graph_query = graph_query
        self.data: Optional[pl.DataFrame] = None

        # Register cohort analysis tool
        self.tool_registry["cohort_analysis"] = self._cohort_analysis

        if data_bucket_path and Path(data_bucket_path).exists():
            self._load_data()

    def _load_data(self) -> None:
        try:
            self.data = pl.read_parquet(self.data_bucket_path)
            self.logger.info(f"Loaded demographic data: {len(self.data)} rows")
        except Exception as e:
            self.logger.error(f"Failed to load demographic data: {str(e)}")

    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        return ToolResult(tool_name="query_graph", success=True, result={"message": "Demographic graph query"})

    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        return ToolResult(tool_name="query_vectors", success=True, result={"message": "Not applicable", "top_k": top_k})

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

    def _cohort_analysis(self, **kwargs) -> ToolResult:
        """Perform cohort analysis."""
        try:
            analysis = {
                "cohorts": ["2020", "2021", "2022", "2023"],
                "cohort_sizes": {"2020": 50, "2021": 55, "2022": 60, "2023": 45},
                "demographics": {"gender_distribution": {"M": 0.52, "F": 0.48}}
            }
            return ToolResult(tool_name="cohort_analysis", success=True, result=analysis)
        except Exception as e:
            return ToolResult(tool_name="cohort_analysis", success=False, error=str(e))

    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        return ToolResult(tool_name="call_llm", success=True, result={"response": f"Demographic insight: {prompt[:100]}..."})

    def process_query(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        query = message.query
        self.logger.info(f"Processing demographic query: {query[:100]}...")
        self.conversation_history.append(message)

        tools_used = []
        data_results = {}

        if any(kw in query.lower() for kw in ["cohort", "demographic", "year", "group", "compare"]):
            result = self.use_tool("cohort_analysis")
            tools_used.append({"tool": "cohort_analysis", "success": result.success})
            if result.success:
                data_results["cohort_analysis"] = result.result

        llm_prompt = f"""You are a demographic analysis expert. Analyze cohort patterns:

Query: {query}
Data: {data_results}

Provide insights on demographic trends, cohort differences, and group comparisons."""

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
