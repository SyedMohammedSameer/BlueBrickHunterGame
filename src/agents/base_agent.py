"""Base agent class and communication protocol."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from loguru import logger
import time


class AgentMessage(BaseModel):
    """Message protocol for inter-agent communication."""

    from_agent: str = Field(..., description="Source agent identifier")
    to_agent: str = Field(..., description="Destination agent identifier")
    query: str = Field(..., description="Natural language query")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data payload")
    priority: int = Field(default=3, ge=1, le=5, description="Priority (1=lowest, 5=highest)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message creation time")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared context")
    message_id: str = Field(default="", description="Unique message identifier")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.message_id:
            self.message_id = f"{self.from_agent}_{self.to_agent}_{int(time.time() * 1000)}"


class AgentResponse(BaseModel):
    """Response from an agent."""

    agent_name: str = Field(..., description="Agent name")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Natural language response")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tools used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""

    def __init__(
        self,
        name: str,
        model: str,
        tools: Optional[List[str]] = None,
        data_bucket_path: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            model: LLM model to use
            tools: List of tool names available to this agent
            data_bucket_path: Path to agent-specific data bucket
            max_retries: Maximum number of retries for failed operations
            timeout: Timeout in seconds for operations
        """
        self.name = name
        self.model = model
        self.tools = tools or []
        self.data_bucket_path = data_bucket_path
        self.max_retries = max_retries
        self.timeout = timeout

        # State management
        self.conversation_history: List[AgentMessage] = []
        self.cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "response_times": [],
            "success_count": [],
            "error_count": [],
        }

        # Tool registry
        self.tool_registry: Dict[str, Callable] = {}
        self._register_default_tools()

        # Logger
        self.logger = logger.bind(name=self.name)
        self.logger.info(f"{self.name} initialized with model {self.model}")

    def _register_default_tools(self) -> None:
        """Register default tools available to all agents."""
        self.tool_registry = {
            "query_graph": self._query_graph,
            "query_vectors": self._query_vectors,
            "compute_stats": self._compute_stats,
            "call_llm": self._call_llm,
        }

    @abstractmethod
    def _query_graph(self, query: str, **kwargs) -> ToolResult:
        """
        Query the knowledge graph.

        Args:
            query: Graph query
            **kwargs: Additional arguments

        Returns:
            Tool result
        """
        pass

    @abstractmethod
    def _query_vectors(self, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        """
        Query the vector store.

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            Tool result
        """
        pass

    @abstractmethod
    def _compute_stats(self, data: Any, stat_type: str, **kwargs) -> ToolResult:
        """
        Compute statistics on data.

        Args:
            data: Input data
            stat_type: Type of statistics to compute
            **kwargs: Additional arguments

        Returns:
            Tool result
        """
        pass

    @abstractmethod
    def _call_llm(self, prompt: str, **kwargs) -> ToolResult:
        """
        Call the LLM with a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Returns:
            Tool result
        """
        pass

    def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if tool_name not in self.tool_registry:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name} not found in registry",
            )

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name} not available to {self.name}",
            )

        start_time = time.time()
        try:
            self.logger.debug(f"Using tool: {tool_name}")
            result = self.tool_registry[tool_name](**kwargs)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    @abstractmethod
    def process_query(self, message: AgentMessage) -> AgentResponse:
        """
        Process a query and return a response.

        Args:
            message: Agent message

        Returns:
            Agent response
        """
        pass

    def communicate(self, target_agent: str, query: str, data: Optional[Dict] = None, priority: int = 3) -> AgentMessage:
        """
        Send a message to another agent.

        Args:
            target_agent: Target agent name
            query: Query string
            data: Optional data payload
            priority: Message priority

        Returns:
            Agent message
        """
        message = AgentMessage(
            from_agent=self.name,
            to_agent=target_agent,
            query=query,
            data=data or {},
            priority=priority,
        )

        self.conversation_history.append(message)
        self.logger.info(f"Sent message to {target_agent}: {query[:100]}...")

        return message

    def update_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Update agent cache.

        Args:
            key: Cache key
            value: Cache value
            ttl: Time to live in seconds (None for no expiration)
        """
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl,
        }

    def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL
        if entry["ttl"] is not None:
            age = (datetime.now() - entry["timestamp"]).total_seconds()
            if age > entry["ttl"]:
                del self.cache[key]
                return None

        return entry["value"]

    def record_performance(self, response_time: float, success: bool) -> None:
        """
        Record performance metrics.

        Args:
            response_time: Response time in seconds
            success: Whether the operation was successful
        """
        self.performance_metrics["response_times"].append(response_time)
        if success:
            self.performance_metrics["success_count"].append(1)
        else:
            self.performance_metrics["error_count"].append(1)

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.

        Returns:
            Dictionary of performance metrics
        """
        response_times = self.performance_metrics["response_times"]
        total_success = sum(self.performance_metrics["success_count"])
        total_errors = sum(self.performance_metrics["error_count"])
        total_requests = total_success + total_errors

        return {
            "total_requests": total_requests,
            "success_rate": total_success / total_requests if total_requests > 0 else 0.0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "cache_hit_rate": len(self.cache) / total_requests if total_requests > 0 else 0.0,
        }

    def clear_cache(self) -> None:
        """Clear agent cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def reset(self) -> None:
        """Reset agent state."""
        self.conversation_history.clear()
        self.cache.clear()
        self.performance_metrics = {
            "response_times": [],
            "success_count": [],
            "error_count": [],
        }
        self.logger.info("Agent reset")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model}, tools={self.tools})"
