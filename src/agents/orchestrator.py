"""Orchestrator Agent - Coordinates all specialized agents and synthesizes responses."""

from typing import Dict, List, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import BaseAgent, AgentMessage, AgentResponse, ToolResult
from .spatial_agent import SpatialAgent
from .behavioral_agent import BehavioralAgent
from .mental_health_agent import MentalHealthAgent
from .temporal_agent import TemporalAgent
from .social_agent import SocialAgent
from .demographic_agent import DemographicAgent


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates all specialized agents.

    This agent:
    - Decomposes complex queries into sub-queries
    - Routes sub-queries to appropriate agents
    - Collects and synthesizes agent responses
    - Handles conflicts and merges insights
    """

    def __init__(
        self,
        name: str = "OrchestratorAgent",
        model: str = "llama3.1:70b",
        agents: Optional[Dict[str, BaseAgent]] = None,
        max_parallel_agents: int = 4,
        **kwargs
    ):
        """
        Initialize Orchestrator Agent.

        Args:
            name: Agent name
            model: LLM model
            agents: Dictionary of specialized agents
            max_parallel_agents: Maximum number of agents to run in parallel
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            model=model,
            tools=["decompose_query", "route_query", "synthesize_responses"],
            **kwargs
        )

        # Register specialized agents
        self.agents: Dict[str, BaseAgent] = agents or {}
        self.max_parallel_agents = max_parallel_agents

        # Register orchestrator tools
        self.tool_registry.update({
            "decompose_query": self._decompose_query,
            "route_query": self._route_query,
            "synthesize_responses": self._synthesize_responses,
        })

        self.logger.info(f"Orchestrator initialized with {len(self.agents)} agents")

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """
        Register a specialized agent.

        Args:
            agent_type: Type of agent (spatial, behavioral, etc.)
            agent: Agent instance
        """
        self.agents[agent_type] = agent
        self.logger.info(f"Registered {agent_type} agent")

    def _decompose_query(self, query: str, **kwargs) -> ToolResult:
        """
        Decompose a complex query into sub-queries for specialized agents.

        Args:
            query: User query

        Returns:
            Tool result with sub-queries
        """
        try:
            sub_queries = {}

            # Keyword-based decomposition (in practice, would use LLM)
            keywords_map = {
                "spatial": ["location", "place", "where", "visit", "mobility", "gps", "travel"],
                "behavioral": ["activity", "sleep", "exercise", "behavior", "screen time", "phone", "app"],
                "mental_health": ["mental health", "phq", "anxiety", "depression", "stress", "mood", "wellbeing"],
                "temporal": ["time", "trend", "longitudinal", "forecast", "predict", "when", "over time"],
                "social": ["social", "interaction", "communication", "friends", "isolation", "network"],
                "demographic": ["cohort", "demographic", "year", "group", "compare", "background"]
            }

            query_lower = query.lower()

            for agent_type, keywords in keywords_map.items():
                if any(kw in query_lower for kw in keywords):
                    sub_queries[agent_type] = query  # Could create more specific sub-queries

            # If no specific keywords, use a general approach
            if not sub_queries:
                # Default to mental health and behavioral for general queries
                sub_queries["mental_health"] = query
                sub_queries["behavioral"] = query

            return ToolResult(
                tool_name="decompose_query",
                success=True,
                result=sub_queries
            )

        except Exception as e:
            return ToolResult(
                tool_name="decompose_query",
                success=False,
                error=str(e)
            )

    def _route_query(self, sub_queries: Dict[str, str], **kwargs) -> ToolResult:
        """
        Route sub-queries to appropriate agents and collect responses.

        Args:
            sub_queries: Dictionary of agent_type -> query

        Returns:
            Tool result with agent responses
        """
        try:
            responses = {}
            parallel = kwargs.get("parallel", True)

            if parallel and len(sub_queries) > 1:
                # Execute agents in parallel
                with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
                    future_to_agent = {
                        executor.submit(self._query_agent, agent_type, query): agent_type
                        for agent_type, query in sub_queries.items()
                        if agent_type in self.agents
                    }

                    for future in as_completed(future_to_agent):
                        agent_type = future_to_agent[future]
                        try:
                            response = future.result(timeout=30)
                            responses[agent_type] = response
                        except Exception as e:
                            self.logger.error(f"Agent {agent_type} failed: {str(e)}")
                            responses[agent_type] = None

            else:
                # Execute agents sequentially
                for agent_type, query in sub_queries.items():
                    if agent_type in self.agents:
                        response = self._query_agent(agent_type, query)
                        responses[agent_type] = response

            return ToolResult(
                tool_name="route_query",
                success=True,
                result=responses
            )

        except Exception as e:
            return ToolResult(
                tool_name="route_query",
                success=False,
                error=str(e)
            )

    def _query_agent(self, agent_type: str, query: str) -> AgentResponse:
        """
        Query a specific agent.

        Args:
            agent_type: Type of agent
            query: Query string

        Returns:
            Agent response
        """
        agent = self.agents.get(agent_type)
        if not agent:
            return AgentResponse(
                agent_name=agent_type,
                query=query,
                response="Agent not available",
                confidence=0.0,
                execution_time=0.0
            )

        message = AgentMessage(
            from_agent=self.name,
            to_agent=agent_type,
            query=query
        )

        return agent.process_query(message)

    def _synthesize_responses(self, responses: Dict[str, AgentResponse], original_query: str, **kwargs) -> ToolResult:
        """
        Synthesize responses from multiple agents into a coherent answer.

        Args:
            responses: Dictionary of agent responses
            original_query: Original user query

        Returns:
            Tool result with synthesized response
        """
        try:
            # Filter out failed responses
            valid_responses = {
                agent_type: resp
                for agent_type, resp in responses.items()
                if resp and resp.confidence > 0.3
            }

            if not valid_responses:
                return ToolResult(
                    tool_name="synthesize_responses",
                    success=False,
                    error="No valid agent responses"
                )

            # Combine insights from all agents
            synthesis = {
                "original_query": original_query,
                "agents_consulted": list(valid_responses.keys()),
                "insights": {},
                "combined_data": {},
                "average_confidence": 0.0
            }

            total_confidence = 0.0
            for agent_type, response in valid_responses.items():
                synthesis["insights"][agent_type] = response.response
                synthesis["combined_data"][agent_type] = response.data
                total_confidence += response.confidence

            synthesis["average_confidence"] = total_confidence / len(valid_responses)

            # Generate synthesized narrative (in practice, would use LLM)
            narrative_parts = [f"Based on analysis from {len(valid_responses)} specialized agents:"]

            for agent_type, response in valid_responses.items():
                agent_name = agent_type.replace("_", " ").title()
                narrative_parts.append(f"\n**{agent_name}**: {response.response}")

            synthesis["synthesized_narrative"] = "\n".join(narrative_parts)

            return ToolResult(
                tool_name="synthesize_responses",
                success=True,
                result=synthesis
            )

        except Exception as e:
            return ToolResult(
                tool_name="synthesize_responses",
                success=False,
                error=str(e)
            )

    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        """Orchestrator doesn't directly query graph."""
        return ToolResult(tool_name="query_graph", success=True, result={"delegated": True})

    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        """Orchestrator doesn't directly query vectors."""
        return ToolResult(tool_name="query_vectors", success=True, result={"delegated": True})

    def _compute_stats(self, data: Any = None, stat_type: str = "summary", **kwargs) -> ToolResult:
        """Orchestrator doesn't directly compute stats."""
        return ToolResult(tool_name="compute_stats", success=True, result={"delegated": True})

    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        """Call LLM for synthesis and reasoning."""
        return ToolResult(
            tool_name="call_llm",
            success=True,
            result={"response": f"Orchestrator synthesis: {prompt[:100]}..."}
        )

    def process_query(self, message: AgentMessage) -> AgentResponse:
        """
        Process query using multi-agent orchestration.

        Args:
            message: Agent message with query

        Returns:
            Orchestrated agent response
        """
        start_time = time.time()
        query = message.query
        self.logger.info(f"Orchestrating query: {query[:100]}...")

        self.conversation_history.append(message)

        # Check cache
        cache_key = f"orchestrator_{hash(query)}"
        cached = self.get_from_cache(cache_key)
        if cached:
            self.logger.info("Returning cached orchestrated result")
            return cached

        tools_used = []

        # Step 1: Decompose query
        decomp_result = self.use_tool("decompose_query", query=query)
        tools_used.append({"tool": "decompose_query", "success": decomp_result.success})

        if not decomp_result.success:
            return AgentResponse(
                agent_name=self.name,
                query=query,
                response="Failed to decompose query",
                confidence=0.0,
                execution_time=time.time() - start_time,
                tool_calls=tools_used
            )

        sub_queries = decomp_result.result

        # Step 2: Route to agents
        route_result = self.use_tool("route_query", sub_queries=sub_queries, parallel=True)
        tools_used.append({"tool": "route_query", "success": route_result.success})

        if not route_result.success:
            return AgentResponse(
                agent_name=self.name,
                query=query,
                response="Failed to route query to agents",
                confidence=0.0,
                execution_time=time.time() - start_time,
                tool_calls=tools_used
            )

        agent_responses = route_result.result

        # Step 3: Synthesize responses
        synth_result = self.use_tool("synthesize_responses", responses=agent_responses, original_query=query)
        tools_used.append({"tool": "synthesize_responses", "success": synth_result.success})

        if not synth_result.success:
            return AgentResponse(
                agent_name=self.name,
                query=query,
                response="Failed to synthesize agent responses",
                confidence=0.0,
                execution_time=time.time() - start_time,
                tool_calls=tools_used
            )

        synthesis = synth_result.result

        # Create final response
        response = AgentResponse(
            agent_name=self.name,
            query=query,
            response=synthesis["synthesized_narrative"],
            data=synthesis["combined_data"],
            confidence=synthesis["average_confidence"],
            execution_time=time.time() - start_time,
            tool_calls=tools_used,
            metadata={
                "agents_consulted": synthesis["agents_consulted"],
                "insights": synthesis["insights"]
            }
        )

        # Cache result
        self.update_cache(cache_key, response, ttl=3600)

        # Record performance
        self.record_performance(response.execution_time, synth_result.success)

        self.logger.info(f"Orchestration complete: {len(synthesis['agents_consulted'])} agents, {response.confidence:.2f} confidence")

        return response

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all agents in the system.

        Returns:
            Dictionary with system status
        """
        status = {
            "orchestrator": {
                "name": self.name,
                "performance": self.get_performance_stats()
            },
            "agents": {}
        }

        for agent_type, agent in self.agents.items():
            status["agents"][agent_type] = {
                "name": agent.name,
                "model": agent.model,
                "performance": agent.get_performance_stats()
            }

        return status
