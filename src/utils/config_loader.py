"""Configuration loading and management."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    models: Dict[str, str] = Field(default_factory=dict)
    embedding: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: int = 4096


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph configuration."""
    backend: str = "networkx"
    neo4j: Dict[str, str] = Field(default_factory=dict)
    node_types: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    backend: str = "chromadb"
    path: str = "./data/chromadb"
    collections: list[str] = Field(default_factory=list)
    similarity_metric: str = "cosine"
    top_k: int = 10


class DataConfig(BaseModel):
    """Data configuration."""
    raw_path: str = "./data/raw"
    processed_path: str = "./data/processed"
    agent_buckets_path: str = "./data/agent_buckets"
    kaggle_dataset: str = "subigyanepal/college-experience-dataset"


class AgentConfig(BaseModel):
    """Individual agent configuration."""
    model: str
    tools: list[str] = Field(default_factory=list)
    max_retries: int = 3
    timeout: int = 30


class AgentsConfig(BaseModel):
    """All agents configuration."""
    orchestrator: AgentConfig
    spatial: AgentConfig
    behavioral: AgentConfig
    mental_health: AgentConfig
    temporal: AgentConfig
    social: AgentConfig
    demographic: AgentConfig


class AnalyticsConfig(BaseModel):
    """Analytics configuration."""
    causal_inference: Dict[str, Any] = Field(default_factory=dict)
    prediction: Dict[str, Any] = Field(default_factory=dict)
    anomaly_detection: Dict[str, Any] = Field(default_factory=dict)
    clustering: Dict[str, Any] = Field(default_factory=dict)


class InterfaceConfig(BaseModel):
    """Interface configuration."""
    gradio: Dict[str, Any] = Field(default_factory=dict)
    tabs: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "./logs/system.log"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    rotation: str = "100 MB"
    retention: str = "30 days"


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    batch_size: int = 64
    num_workers: int = 4
    cache_enabled: bool = True
    cache_size: int = 1000
    gpu_acceleration: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: Dict[str, list[str]] = Field(default_factory=dict)
    benchmark_queries_path: str = "./config/benchmark_queries.json"


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig
    knowledge_graph: KnowledgeGraphConfig
    vector_store: VectorStoreConfig
    data: DataConfig
    agents: AgentsConfig
    analytics: AnalyticsConfig
    interface: InterfaceConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    evaluation: EvaluationConfig


class ConfigLoader:
    """Configuration loader with environment variable support."""

    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
            env_file: Path to .env file
        """
        # Load environment variables
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        else:
            load_dotenv()  # Try to load from default .env

        # Load YAML configuration
        if config_path is None:
            config_path = "config/config.yaml"

        self.config_path = Path(config_path)
        self.config_data = self._load_yaml()

        # Override with environment variables
        self._override_from_env()

        # Parse into Pydantic model
        self.config = Config(**self.config_data)

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _override_from_env(self) -> None:
        """Override configuration values from environment variables."""
        # Ollama configuration
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.config_data["llm"]["base_url"] = ollama_url

        # Neo4j configuration
        if neo4j_uri := os.getenv("NEO4J_URI"):
            self.config_data["knowledge_graph"]["neo4j"]["uri"] = neo4j_uri
        if neo4j_user := os.getenv("NEO4J_USERNAME"):
            self.config_data["knowledge_graph"]["neo4j"]["username"] = neo4j_user
        if neo4j_pass := os.getenv("NEO4J_PASSWORD"):
            self.config_data["knowledge_graph"]["neo4j"]["password"] = neo4j_pass

        # ChromaDB configuration
        if chromadb_path := os.getenv("CHROMADB_PATH"):
            self.config_data["vector_store"]["path"] = chromadb_path

        # Logging configuration
        if log_level := os.getenv("LOG_LEVEL"):
            self.config_data["logging"]["level"] = log_level
        if log_file := os.getenv("LOG_FILE"):
            self.config_data["logging"]["file"] = log_file

        # Performance configuration
        if batch_size := os.getenv("BATCH_SIZE"):
            self.config_data["performance"]["batch_size"] = int(batch_size)
        if num_workers := os.getenv("NUM_WORKERS"):
            self.config_data["performance"]["num_workers"] = int(num_workers)
        if gpu_enabled := os.getenv("GPU_ENABLED"):
            self.config_data["performance"]["gpu_acceleration"] = gpu_enabled.lower() == "true"

        # Gradio configuration
        if gradio_server := os.getenv("GRADIO_SERVER_NAME"):
            self.config_data["interface"]["gradio"]["server_name"] = gradio_server
        if gradio_port := os.getenv("GRADIO_SERVER_PORT"):
            self.config_data["interface"]["gradio"]["server_port"] = int(gradio_port)
        if gradio_share := os.getenv("GRADIO_SHARE"):
            self.config_data["interface"]["gradio"]["share"] = gradio_share.lower() == "true"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'llm.provider')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


def load_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> Config:
    """
    Load configuration from file and environment.

    Args:
        config_path: Path to YAML configuration file
        env_file: Path to .env file

    Returns:
        Configuration object
    """
    loader = ConfigLoader(config_path, env_file)
    return loader.config
