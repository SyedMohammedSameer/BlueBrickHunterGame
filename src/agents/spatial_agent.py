"""Spatial Agent - Handles GPS, location semantics, and mobility patterns."""

from typing import Dict, Any, Optional
from pathlib import Path
import polars as pl
import numpy as np

from .base_agent import BaseAgent, AgentMessage, AgentResponse, ToolResult
from ..knowledge_graph.graph_query import GraphQuery
from ..knowledge_graph.schema import NodeType, RelationshipType
from ..embeddings.vector_store import VectorStoreManager


class SpatialAgent(BaseAgent):
    """Agent specialized in spatial data analysis."""

    def __init__(
        self,
        name: str = "SpatialAgent",
        model: str = "llama3.1:70b",
        data_bucket_path: Optional[str] = None,
        graph_query: Optional[GraphQuery] = None,
        vector_store: Optional[VectorStoreManager] = None,
        **kwargs
    ):
        """
        Initialize Spatial Agent.

        Args:
            name: Agent name
            model: LLM model
            data_bucket_path: Path to spatial data bucket
            graph_query: Graph query interface
            vector_store: Vector store manager
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(
            name=name,
            model=model,
            tools=["query_graph", "query_vectors", "compute_stats"],
            data_bucket_path=data_bucket_path,
            **kwargs
        )

        self.graph_query = graph_query
        self.vector_store = vector_store
        self.data: Optional[pl.DataFrame] = None

        # Load data if path provided
        if data_bucket_path and Path(data_bucket_path).exists():
            self._load_data()

    def _load_data(self) -> None:
        """Load spatial data from bucket."""
        try:
            self.data = pl.read_parquet(self.data_bucket_path)
            self.logger.info(f"Loaded spatial data: {len(self.data)} rows")
        except Exception as e:
            self.logger.error(f"Failed to load spatial data: {str(e)}")

    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        """Query knowledge graph for location data."""
        if not self.graph_query:
            return ToolResult(
                tool_name="query_graph",
                success=False,
                error="Graph query interface not available"
            )

        try:
            # Example: Get all location nodes
            if "locations" in query.lower():
                locations = self.graph_query.get_nodes_by_type(NodeType.LOCATION)
                return ToolResult(
                    tool_name="query_graph",
                    success=True,
                    result=locations
                )

            # Example: Get student location visits
            if "student" in query.lower() and "visits" in query.lower():
                student_id = kwargs.get("student_id")
                if student_id:
                    node_id = f"student_{student_id}"
                    visits = self.graph_query.get_relationships(
                        source_id=node_id,
                        rel_type=RelationshipType.VISITED
                    )
                    return ToolResult(
                        tool_name="query_graph",
                        success=True,
                        result=visits
                    )

            return ToolResult(
                tool_name="query_graph",
                success=True,
                result={"message": "Query processed"}
            )

        except Exception as e:
            return ToolResult(
                tool_name="query_graph",
                success=False,
                error=str(e)
            )

    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        """Query vector store for similar location patterns."""
        if not self.vector_store:
            return ToolResult(
                tool_name="query_vectors",
                success=False,
                error="Vector store not available"
            )

        try:
            # In practice, would embed query and search
            # For now, return placeholder
            return ToolResult(
                tool_name="query_vectors",
                success=True,
                result={"message": "Vector query processed", "top_k": top_k}
            )

        except Exception as e:
            return ToolResult(
                tool_name="query_vectors",
                success=False,
                error=str(e)
            )

    def _compute_stats(self, data: Any = None, stat_type: str = "summary", **kwargs) -> ToolResult:
        """Compute statistics on spatial data."""
        try:
            if data is None and self.data is not None:
                data = self.data

            if data is None:
                return ToolResult(
                    tool_name="compute_stats",
                    success=False,
                    error="No data available"
                )

            stats = {}

            if stat_type == "summary":
                # Basic summary statistics
                stats = {
                    "num_records": len(data),
                    "num_columns": len(data.columns),
                    "columns": data.columns
                }

            elif stat_type == "location_frequency":
                # Location visit frequency
                location_cols = [col for col in data.columns if "location" in col.lower() or "place" in col.lower()]
                if location_cols:
                    freq = data[location_cols[0]].value_counts()
                    stats["location_frequency"] = freq.to_dict()

            return ToolResult(
                tool_name="compute_stats",
                success=True,
                result=stats
            )

        except Exception as e:
            return ToolResult(
                tool_name="compute_stats",
                success=False,
                error=str(e)
            )

    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        """Call LLM for spatial analysis insights."""
        try:
            # Placeholder for actual LLM call to Ollama
            # In practice, would use requests to call Ollama API
            response = f"Spatial analysis insight for: {prompt[:100]}..."

            return ToolResult(
                tool_name="call_llm",
                success=True,
                result={"response": response}
            )

        except Exception as e:
            return ToolResult(
                tool_name="call_llm",
                success=False,
                error=str(e)
            )

    def process_query(self, message: AgentMessage) -> AgentResponse:
        """Process spatial query and return response."""
        import time
        start_time = time.time()

        query = message.query
        self.logger.info(f"Processing spatial query: {query[:100]}...")

        # Add to conversation history
        self.conversation_history.append(message)

        # Check cache
        cache_key = f"spatial_{hash(query)}"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.logger.info("Returning cached result")
            return cached_result

        # Determine which tools to use based on query
        tools_used = []
        data_results = {}

        # Check if query asks for locations
        if any(kw in query.lower() for kw in ["location", "place", "where", "visit"]):
            result = self.use_tool("query_graph", query=query)
            tools_used.append({"tool": "query_graph", "success": result.success})
            if result.success:
                data_results["locations"] = result.result

        # Check if query asks for statistics
        if any(kw in query.lower() for kw in ["how many", "frequency", "count", "statistics"]):
            result = self.use_tool("compute_stats", stat_type="location_frequency")
            tools_used.append({"tool": "compute_stats", "success": result.success})
            if result.success:
                data_results["statistics"] = result.result

        # Generate response using LLM
        llm_prompt = f"""You are a spatial analysis expert. Based on the following query and data, provide insights:

Query: {query}

Data: {data_results}

Provide a concise, informative response focusing on spatial patterns, mobility, and location insights."""

        llm_result = self.use_tool("call_llm", prompt=llm_prompt)
        tools_used.append({"tool": "call_llm", "success": llm_result.success})

        response_text = llm_result.result.get("response", "Unable to generate response") if llm_result.success else "Error processing query"

        # Create response
        response = AgentResponse(
            agent_name=self.name,
            query=query,
            response=response_text,
            data=data_results,
            confidence=0.85 if llm_result.success else 0.3,
            execution_time=time.time() - start_time,
            tool_calls=tools_used
        )

        # Cache result
        self.update_cache(cache_key, response, ttl=3600)

        # Record performance
        self.record_performance(response.execution_time, llm_result.success)

        return response

    def get_location_correlations(self, student_id: str) -> Dict[str, Any]:
        """
        Get location correlations with mental health for a student.

        Args:
            student_id: Student identifier

        Returns:
            Dictionary with correlation data
        """
        if not self.graph_query:
            return {}

        # Get student's location visits
        visits = self.graph_query.get_relationships(
            source_id=f"student_{student_id}",
            rel_type=RelationshipType.VISITED
        )

        # Get mental health states
        mh_states = self.graph_query.get_relationships(
            source_id=f"student_{student_id}",
            rel_type=RelationshipType.EXPERIENCED
        )

        return {
            "student_id": student_id,
            "num_locations_visited": len(visits),
            "num_mental_health_assessments": len(mh_states),
            "visits": visits[:10],  # Top 10
            "mental_health_states": mh_states[:10]
        }
