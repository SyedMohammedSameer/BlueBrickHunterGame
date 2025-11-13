"""Knowledge graph construction from data."""

import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import polars as pl
from loguru import logger

from .schema import (
    Node,
    StudentNode,
    LocationNode,
    ActivityNode,
    MentalHealthStateNode,
    TemporalEventNode,
    DemographicNode,
    Relationship,
    RelationshipType,
)


class KnowledgeGraphBuilder:
    """Build knowledge graph from college experience data."""

    def __init__(self, backend: str = "networkx"):
        """
        Initialize graph builder.

        Args:
            backend: Graph backend ('networkx' or 'neo4j')
        """
        self.backend = backend
        self.graph = nx.MultiDiGraph() if backend == "networkx" else None
        self.nodes: Dict[str, Node] = {}
        self.relationships: List[Relationship] = []
        self.logger = logger.bind(name=__name__)

        self.logger.info(f"KnowledgeGraphBuilder initialized with {backend} backend")

    def add_node(self, node: Node) -> None:
        """
        Add a node to the knowledge graph.

        Args:
            node: Node to add
        """
        if node.id in self.nodes:
            # Update existing node
            self.nodes[node.id].properties.update(node.properties)
        else:
            self.nodes[node.id] = node

        if self.backend == "networkx":
            self.graph.add_node(
                node.id,
                type=node.type.value,
                **node.properties
            )

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the knowledge graph.

        Args:
            relationship: Relationship to add
        """
        self.relationships.append(relationship)

        if self.backend == "networkx":
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                type=relationship.rel_type.value,
                weight=relationship.weight,
                timestamp=relationship.timestamp,
                **relationship.properties
            )

    def build_from_data(
        self,
        spatial_df: Optional[pl.DataFrame] = None,
        behavioral_df: Optional[pl.DataFrame] = None,
        mental_health_df: Optional[pl.DataFrame] = None,
        temporal_df: Optional[pl.DataFrame] = None,
        social_df: Optional[pl.DataFrame] = None,
        demographic_df: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Build knowledge graph from agent data buckets.

        Args:
            spatial_df: Spatial data
            behavioral_df: Behavioral data
            mental_health_df: Mental health data
            temporal_df: Temporal data
            social_df: Social data
            demographic_df: Demographic data
        """
        self.logger.info("Building knowledge graph from data...")

        # Extract unique students
        student_ids = set()
        for df in [spatial_df, behavioral_df, mental_health_df, temporal_df, social_df, demographic_df]:
            if df is not None:
                for id_col in ["uid", "student_id", "user_id"]:
                    if id_col in df.columns:
                        student_ids.update(df[id_col].unique().to_list())
                        break

        # Create student nodes
        self.logger.info(f"Creating {len(student_ids)} student nodes...")
        for student_id in student_ids:
            if student_id is not None:
                self.add_node(StudentNode(student_id=str(student_id)))

        # Process spatial data
        if spatial_df is not None:
            self._process_spatial_data(spatial_df)

        # Process behavioral data
        if behavioral_df is not None:
            self._process_behavioral_data(behavioral_df)

        # Process mental health data
        if mental_health_df is not None:
            self._process_mental_health_data(mental_health_df)

        # Process temporal data
        if temporal_df is not None:
            self._process_temporal_data(temporal_df)

        # Process demographic data
        if demographic_df is not None:
            self._process_demographic_data(demographic_df)

        self.logger.info(f"Knowledge graph built: {len(self.nodes)} nodes, {len(self.relationships)} relationships")

    def _process_spatial_data(self, df: pl.DataFrame) -> None:
        """Process spatial data and create location nodes and relationships."""
        self.logger.info("Processing spatial data...")

        # Identify location columns
        location_cols = [col for col in df.columns if any(kw in col.lower() for kw in ["location", "place", "semantic", "lat", "lon"])]

        if not location_cols:
            self.logger.warning("No location columns found in spatial data")
            return

        # Group by student and location to create VISITED relationships
        id_col = next((col for col in ["uid", "student_id"] if col in df.columns), None)
        if not id_col:
            return

        # Sample processing (adapt based on actual columns)
        for row in df.iter_rows(named=True):
            student_id = str(row.get(id_col))

            # Create location node if location info available
            location_id = row.get("location", row.get("place", "unknown"))
            if location_id and location_id != "unknown":
                location_node = LocationNode(
                    location_id=str(location_id),
                    semantic_label=str(location_id),
                    visit_count=1
                )
                self.add_node(location_node)

                # Create VISITED relationship
                rel = Relationship(
                    source_id=f"student_{student_id}",
                    target_id=f"location_{location_id}",
                    rel_type=RelationshipType.VISITED,
                    timestamp=row.get("timestamp"),
                    properties={"duration": row.get("duration")}
                )
                self.add_relationship(rel)

    def _process_behavioral_data(self, df: pl.DataFrame) -> None:
        """Process behavioral data and create activity nodes and relationships."""
        self.logger.info("Processing behavioral data...")

        id_col = next((col for col in ["uid", "student_id"] if col in df.columns), None)
        if not id_col:
            return

        # Create activity nodes
        activity_id_counter = 0
        for row in df.iter_rows(named=True):
            student_id = str(row.get(id_col))
            activity_type = row.get("activity", row.get("activity_type", "unknown"))

            if activity_type != "unknown":
                activity_node = ActivityNode(
                    activity_id=str(activity_id_counter),
                    activity_type=str(activity_type),
                    duration=row.get("duration"),
                    timestamp=row.get("timestamp")
                )
                self.add_node(activity_node)

                # Create PERFORMED relationship
                rel = Relationship(
                    source_id=f"student_{student_id}",
                    target_id=f"activity_{activity_id_counter}",
                    rel_type=RelationshipType.PERFORMED,
                    timestamp=row.get("timestamp")
                )
                self.add_relationship(rel)

                activity_id_counter += 1

    def _process_mental_health_data(self, df: pl.DataFrame) -> None:
        """Process mental health data and create mental health state nodes."""
        self.logger.info("Processing mental health data...")

        id_col = next((col for col in ["uid", "student_id"] if col in df.columns), None)
        if not id_col:
            return

        # Create mental health state nodes
        state_id_counter = 0
        for row in df.iter_rows(named=True):
            student_id = str(row.get(id_col))

            # Extract PHQ4 or related scores
            phq4_score = row.get("phq4", row.get("phq4_score"))
            anxiety_score = row.get("anxiety", row.get("anxiety_score"))
            depression_score = row.get("depression", row.get("depression_score"))

            if phq4_score is not None or anxiety_score is not None or depression_score is not None:
                state_node = MentalHealthStateNode(
                    state_id=str(state_id_counter),
                    phq4_score=phq4_score,
                    anxiety_score=anxiety_score,
                    depression_score=depression_score,
                    timestamp=row.get("timestamp")
                )
                self.add_node(state_node)

                # Create EXPERIENCED relationship
                rel = Relationship(
                    source_id=f"student_{student_id}",
                    target_id=f"mental_health_{state_id_counter}",
                    rel_type=RelationshipType.EXPERIENCED,
                    timestamp=row.get("timestamp")
                )
                self.add_relationship(rel)

                state_id_counter += 1

    def _process_temporal_data(self, df: pl.DataFrame) -> None:
        """Process temporal data and create temporal event nodes."""
        self.logger.info("Processing temporal data...")

        # Identify temporal events (exam periods, holidays, etc.)
        # This is a placeholder - actual implementation depends on data structure
        pass

    def _process_demographic_data(self, df: pl.DataFrame) -> None:
        """Process demographic data and update student nodes."""
        self.logger.info("Processing demographic data...")

        id_col = next((col for col in ["uid", "student_id"] if col in df.columns), None)
        if not id_col:
            return

        for row in df.iter_rows(named=True):
            student_id = str(row.get(id_col))
            student_node_id = f"student_{student_id}"

            if student_node_id in self.nodes:
                # Update student node with demographic info
                self.nodes[student_node_id].properties.update({
                    "cohort": row.get("cohort"),
                    "academic_year": row.get("year"),
                    "demographics": {k: v for k, v in row.items() if k not in [id_col, "cohort", "year"]}
                })

    def compute_correlations(self) -> None:
        """Compute correlations between locations and mental health states."""
        self.logger.info("Computing correlations...")

        # Find students who have both location visits and mental health assessments
        # Compute correlations and add CORRELATES_WITH relationships
        # This is a simplified version - full implementation would use statistical methods

        location_nodes = {nid: n for nid, n in self.nodes.items() if "location" in nid}
        mh_nodes = {nid: n for nid, n in self.nodes.items() if "mental_health" in nid}

        # Placeholder for correlation computation
        # In practice, would analyze co-occurrence patterns and compute statistics

    def save_graph(self, output_path: Union[str, Path], format: str = "graphml") -> None:
        """
        Save knowledge graph to file.

        Args:
            output_path: Output file path
            format: File format ('graphml', 'gexf', 'json')
        """
        if self.backend != "networkx":
            self.logger.warning("Save only implemented for NetworkX backend")
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "graphml":
            nx.write_graphml(self.graph, output_file)
        elif format == "gexf":
            nx.write_gexf(self.graph, output_file)
        elif format == "json":
            import json
            data = nx.node_link_data(self.graph)
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Knowledge graph saved to {output_file}")

    def load_graph(self, input_path: Union[str, Path], format: str = "graphml") -> None:
        """
        Load knowledge graph from file.

        Args:
            input_path: Input file path
            format: File format ('graphml', 'gexf', 'json')
        """
        if self.backend != "networkx":
            self.logger.warning("Load only implemented for NetworkX backend")
            return

        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Graph file not found: {input_file}")

        if format == "graphml":
            self.graph = nx.read_graphml(input_file)
        elif format == "gexf":
            self.graph = nx.read_gexf(input_file)
        elif format == "json":
            import json
            with open(input_file, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

        self.logger.info(f"Knowledge graph loaded from {input_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary of graph statistics
        """
        if self.backend != "networkx":
            return {}

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_components": nx.number_weakly_connected_components(self.graph),
        }
