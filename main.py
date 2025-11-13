#!/usr/bin/env python3
"""
Multi-Agent College Mental Health Analysis System - Main Application

This script demonstrates the complete multi-agent system for analyzing
college student mental health data from the Dartmouth StudentLife dataset.

Usage:
    python main.py --mode demo
    python main.py --mode query --question "What locations correlate with better mental health?"
    python main.py --mode build-graph
    python main.py --mode status
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.config_loader import load_config
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStoreManager
from src.agents.orchestrator import OrchestratorAgent
from src.agents.spatial_agent import SpatialAgent
from src.agents.behavioral_agent import BehavioralAgent
from src.agents.mental_health_agent import MentalHealthAgent
from src.agents.temporal_agent import TemporalAgent
from src.agents.social_agent import SocialAgent
from src.agents.demographic_agent import DemographicAgent
from src.agents.base_agent import AgentMessage


class MultiAgentSystem:
    """Main multi-agent system orchestrator."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the multi-agent system.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)

        # Initialize components
        self.graph_builder = None
        self.graph_query = None
        self.embedding_generator = None
        self.vector_store = None
        self.orchestrator = None

        self.logger.info("Multi-Agent System initialized")

    def setup(self) -> None:
        """Set up all system components."""
        self.logger.info("Setting up system components...")

        # Initialize knowledge graph
        self.graph_builder = KnowledgeGraphBuilder(
            backend=self.config.knowledge_graph.backend
        )

        # Try to load existing graph
        graph_path = Path("data/knowledge_graph.graphml")
        if graph_path.exists():
            self.logger.info("Loading existing knowledge graph...")
            self.graph_builder.load_graph(graph_path)
        else:
            self.logger.warning("No knowledge graph found. Run --mode build-graph to create one.")

        self.graph_query = GraphQuery(self.graph_builder.graph)

        # Initialize embeddings
        self.embedding_generator = EmbeddingGenerator(
            base_url=self.config.llm.base_url,
            model=self.config.llm.embedding.model,
            dimension=self.config.llm.embedding.dimension
        )

        # Initialize vector store
        self.vector_store = VectorStoreManager(
            persist_directory=self.config.vector_store.path,
            collection_name="student_behaviors"
        )

        # Initialize agents
        self._initialize_agents()

        self.logger.info("System setup complete")

    def _initialize_agents(self) -> None:
        """Initialize all specialized agents and orchestrator."""
        self.logger.info("Initializing agents...")

        # Agent data bucket paths
        bucket_path = Path(self.config.data.agent_buckets_path)

        # Create specialized agents
        spatial = SpatialAgent(
            model=self.config.agents.spatial.model,
            data_bucket_path=str(bucket_path / "spatial_data.parquet"),
            graph_query=self.graph_query,
            vector_store=self.vector_store
        )

        behavioral = BehavioralAgent(
            model=self.config.agents.behavioral.model,
            data_bucket_path=str(bucket_path / "behavioral_data.parquet"),
            graph_query=self.graph_query,
            vector_store=self.vector_store
        )

        mental_health = MentalHealthAgent(
            model=self.config.agents.mental_health.model,
            data_bucket_path=str(bucket_path / "mental_health_data.parquet"),
            graph_query=self.graph_query,
            vector_store=self.vector_store
        )

        temporal = TemporalAgent(
            model=self.config.agents.temporal.model,
            data_bucket_path=str(bucket_path / "temporal_data.parquet"),
            graph_query=self.graph_query,
            vector_store=self.vector_store
        )

        social = SocialAgent(
            model=self.config.agents.social.model,
            data_bucket_path=str(bucket_path / "social_data.parquet"),
            graph_query=self.graph_query,
            vector_store=self.vector_store
        )

        demographic = DemographicAgent(
            model=self.config.agents.demographic.model,
            data_bucket_path=str(bucket_path / "demographic_data.parquet"),
            graph_query=self.graph_query
        )

        # Create orchestrator with all agents
        agents = {
            "spatial": spatial,
            "behavioral": behavioral,
            "mental_health": mental_health,
            "temporal": temporal,
            "social": social,
            "demographic": demographic
        }

        self.orchestrator = OrchestratorAgent(
            model=self.config.agents.orchestrator.model,
            agents=agents,
            max_parallel_agents=4
        )

        self.logger.info(f"Initialized {len(agents) + 1} agents")

    def build_knowledge_graph(self) -> None:
        """Build knowledge graph from data."""
        self.logger.info("Building knowledge graph from data...")

        bucket_path = Path(self.config.data.agent_buckets_path)

        # Check if data buckets exist
        if not bucket_path.exists():
            self.logger.error(f"Data buckets not found at {bucket_path}")
            self.logger.info("Please run the data pipeline first: python src/data_pipeline.py")
            return

        # Load data
        import polars as pl

        data_files = {
            "spatial": bucket_path / "spatial_data.parquet",
            "behavioral": bucket_path / "behavioral_data.parquet",
            "mental_health": bucket_path / "mental_health_data.parquet",
            "temporal": bucket_path / "temporal_data.parquet",
            "social": bucket_path / "social_data.parquet",
            "demographic": bucket_path / "demographic_data.parquet"
        }

        loaded_data = {}
        for name, path in data_files.items():
            if path.exists():
                loaded_data[f"{name}_df"] = pl.read_parquet(path)
                self.logger.info(f"Loaded {name} data: {len(loaded_data[f'{name}_df'])} rows")
            else:
                self.logger.warning(f"{name} data not found at {path}")

        # Build graph
        self.graph_builder.build_from_data(**loaded_data)

        # Save graph
        output_path = Path("data/knowledge_graph.graphml")
        self.graph_builder.save_graph(output_path)

        # Print statistics
        stats = self.graph_builder.get_statistics()
        self.logger.info("Knowledge Graph Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")

    def query(self, question: str) -> None:
        """
        Query the multi-agent system.

        Args:
            question: User question
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Query: {question}")
        self.logger.info(f"{'='*60}")

        # Create message
        message = AgentMessage(
            from_agent="user",
            to_agent="orchestrator",
            query=question
        )

        # Process query
        response = self.orchestrator.process_query(message)

        # Display response
        print(f"\n{'='*60}")
        print(f"RESPONSE FROM {response.agent_name}")
        print(f"{'='*60}")
        print(f"\n{response.response}\n")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Execution Time: {response.execution_time:.2f}s")

        if response.metadata:
            print(f"\nAgents Consulted: {', '.join(response.metadata.get('agents_consulted', []))}")

        if response.data:
            print(f"\nData Available: {list(response.data.keys())}")

        print(f"\n{'='*60}\n")

    def demo(self) -> None:
        """Run demonstration queries."""
        self.logger.info("\n" + "="*60)
        self.logger.info("RUNNING DEMONSTRATION QUERIES")
        self.logger.info("="*60 + "\n")

        demo_queries = [
            "What locations are most frequently visited by students?",
            "How do activity patterns relate to mental health scores?",
            "What are the temporal trends in PHQ4 scores?",
            "Which students show signs of social isolation?",
            "Compare mental health across different cohorts",
            "What behavioral patterns predict better mental health outcomes?"
        ]

        for i, question in enumerate(demo_queries, 1):
            print(f"\n{'#'*60}")
            print(f"DEMO QUERY {i}/{len(demo_queries)}")
            print(f"{'#'*60}\n")
            self.query(question)
            print("\n")

    def status(self) -> None:
        """Display system status."""
        print(f"\n{'='*60}")
        print("MULTI-AGENT SYSTEM STATUS")
        print(f"{'='*60}\n")

        if self.orchestrator:
            status = self.orchestrator.get_system_status()

            print("Orchestrator:")
            print(f"  Name: {status['orchestrator']['name']}")
            perf = status['orchestrator']['performance']
            print(f"  Total Requests: {perf['total_requests']}")
            print(f"  Success Rate: {perf['success_rate']:.2%}")
            print(f"  Avg Response Time: {perf['avg_response_time']:.2f}s")

            print("\nSpecialized Agents:")
            for agent_type, agent_status in status['agents'].items():
                print(f"\n  {agent_type.title()}:")
                print(f"    Model: {agent_status['model']}")
                perf = agent_status['performance']
                print(f"    Requests: {perf['total_requests']}")
                print(f"    Success Rate: {perf['success_rate']:.2%}")

        # Knowledge Graph Status
        if self.graph_builder:
            print("\nKnowledge Graph:")
            stats = self.graph_builder.get_statistics()
            for key, value in stats.items():
                print(f"  {key}: {value}")

        # Vector Store Status
        if self.vector_store:
            print("\nVector Store:")
            print(f"  Embeddings Count: {self.vector_store.count()}")

        print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent College Mental Health Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo
  python main.py --mode query --question "What locations correlate with mental health?"
  python main.py --mode build-graph
  python main.py --mode status
        """
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "query", "build-graph", "status"],
        default="demo",
        help="Operation mode"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Query question (for query mode)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_file="logs/main.log", level=args.log_level)

    logger = get_logger(__name__)
    logger.info("Starting Multi-Agent College Mental Health Analysis System")

    # Initialize system
    system = MultiAgentSystem(config_path=args.config)

    if args.mode == "build-graph":
        # Build knowledge graph
        system.setup()
        system.build_knowledge_graph()

    elif args.mode == "status":
        # Show status
        system.setup()
        system.status()

    elif args.mode == "query":
        # Query mode
        if not args.question:
            logger.error("--question required for query mode")
            return

        system.setup()
        system.query(args.question)

    elif args.mode == "demo":
        # Demo mode
        system.setup()
        system.demo()

    logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
