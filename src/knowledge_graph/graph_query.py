"""Knowledge graph query interface."""

import networkx as nx
from typing import List, Dict, Any, Optional, Union, Set
from loguru import logger

from .schema import NodeType, RelationshipType


class GraphQuery:
    """Query interface for the knowledge graph."""

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize graph query interface.

        Args:
            graph: NetworkX graph instance
        """
        self.graph = graph
        self.logger = logger.bind(name=__name__)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node data or None if not found
        """
        if node_id not in self.graph:
            return None

        return {
            "id": node_id,
            **self.graph.nodes[node_id]
        }

    def get_nodes_by_type(self, node_type: Union[str, NodeType]) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: Node type

        Returns:
            List of nodes
        """
        if isinstance(node_type, NodeType):
            node_type = node_type.value

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == node_type:
                nodes.append({"id": node_id, **data})

        return nodes

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[Union[str, RelationshipType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relationships matching criteria.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type

        Returns:
            List of relationships
        """
        if isinstance(rel_type, RelationshipType):
            rel_type = rel_type.value

        relationships = []

        if source_id and target_id:
            # Get specific edge
            if self.graph.has_edge(source_id, target_id):
                for key, data in self.graph[source_id][target_id].items():
                    if rel_type is None or data.get("type") == rel_type:
                        relationships.append({
                            "source": source_id,
                            "target": target_id,
                            **data
                        })
        elif source_id:
            # Get all outgoing edges from source
            for target in self.graph.successors(source_id):
                for key, data in self.graph[source_id][target].items():
                    if rel_type is None or data.get("type") == rel_type:
                        relationships.append({
                            "source": source_id,
                            "target": target,
                            **data
                        })
        elif target_id:
            # Get all incoming edges to target
            for source in self.graph.predecessors(target_id):
                for key, data in self.graph[source][target_id].items():
                    if rel_type is None or data.get("type") == rel_type:
                        relationships.append({
                            "source": source,
                            "target": target_id,
                            **data
                        })
        else:
            # Get all edges
            for source, target, data in self.graph.edges(data=True):
                if rel_type is None or data.get("type") == rel_type:
                    relationships.append({
                        "source": source,
                        "target": target,
                        **data
                    })

        return relationships

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        rel_type: Optional[Union[str, RelationshipType]] = None
    ) -> List[str]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node identifier
            direction: 'in', 'out', or 'both'
            rel_type: Filter by relationship type

        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []

        if isinstance(rel_type, RelationshipType):
            rel_type = rel_type.value

        neighbors = set()

        if direction in ["out", "both"]:
            for target in self.graph.successors(node_id):
                if rel_type is None:
                    neighbors.add(target)
                else:
                    for key, data in self.graph[node_id][target].items():
                        if data.get("type") == rel_type:
                            neighbors.add(target)
                            break

        if direction in ["in", "both"]:
            for source in self.graph.predecessors(node_id):
                if rel_type is None:
                    neighbors.add(source)
                else:
                    for key, data in self.graph[source][node_id].items():
                        if data.get("type") == rel_type:
                            neighbors.add(source)
                            break

        return list(neighbors)

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        Find all simple paths between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Get shortest path between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: Edge weight attribute for weighted paths

        Returns:
            Shortest path or None if no path exists
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None

        try:
            return nx.shortest_path(
                self.graph,
                source_id,
                target_id,
                weight=weight
            )
        except nx.NetworkXNoPath:
            return None

    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """
        Get node degree (in, out, total).

        Args:
            node_id: Node identifier

        Returns:
            Dictionary with degree information
        """
        if node_id not in self.graph:
            return {"in": 0, "out": 0, "total": 0}

        return {
            "in": self.graph.in_degree(node_id),
            "out": self.graph.out_degree(node_id),
            "total": self.graph.degree(node_id)
        }

    def compute_centrality(
        self,
        centrality_type: str = "pagerank"
    ) -> Dict[str, float]:
        """
        Compute node centrality measures.

        Args:
            centrality_type: Type of centrality ('pagerank', 'betweenness', 'closeness')

        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        if centrality_type == "pagerank":
            return nx.pagerank(self.graph)
        elif centrality_type == "betweenness":
            return nx.betweenness_centrality(self.graph)
        elif centrality_type == "closeness":
            return nx.closeness_centrality(self.graph)
        else:
            self.logger.warning(f"Unknown centrality type: {centrality_type}")
            return {}

    def find_communities(self) -> List[Set[str]]:
        """
        Find communities in the graph using the Louvain method.

        Returns:
            List of communities (sets of node IDs)
        """
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        # Use greedy modularity maximization
        communities = nx.community.greedy_modularity_communities(undirected)

        return [set(community) for community in communities]

    def subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """
        Get subgraph containing specified nodes.

        Args:
            node_ids: List of node IDs

        Returns:
            Subgraph
        """
        return self.graph.subgraph(node_ids).copy()

    def search_nodes(
        self,
        property_name: str,
        property_value: Any,
        node_type: Optional[Union[str, NodeType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes by property value.

        Args:
            property_name: Property name to search
            property_value: Property value to match
            node_type: Optional node type filter

        Returns:
            List of matching nodes
        """
        if isinstance(node_type, NodeType):
            node_type = node_type.value

        matching_nodes = []

        for node_id, data in self.graph.nodes(data=True):
            # Check node type filter
            if node_type and data.get("type") != node_type:
                continue

            # Check property match
            if data.get(property_name) == property_value:
                matching_nodes.append({"id": node_id, **data})

        return matching_nodes

    def aggregate_relationships(
        self,
        source_type: Union[str, NodeType],
        target_type: Union[str, NodeType],
        rel_type: Union[str, RelationshipType]
    ) -> Dict[str, int]:
        """
        Aggregate relationship counts between node types.

        Args:
            source_type: Source node type
            target_type: Target node type
            rel_type: Relationship type

        Returns:
            Dictionary mapping (source_id, target_id) to count
        """
        if isinstance(source_type, NodeType):
            source_type = source_type.value
        if isinstance(target_type, NodeType):
            target_type = target_type.value
        if isinstance(rel_type, RelationshipType):
            rel_type = rel_type.value

        aggregation = {}

        for source, target, data in self.graph.edges(data=True):
            # Check if edge matches criteria
            if (
                self.graph.nodes[source].get("type") == source_type
                and self.graph.nodes[target].get("type") == target_type
                and data.get("type") == rel_type
            ):
                key = (source, target)
                aggregation[key] = aggregation.get(key, 0) + 1

        return aggregation

    def get_student_behavior_summary(self, student_id: str) -> Dict[str, Any]:
        """
        Get comprehensive behavior summary for a student.

        Args:
            student_id: Student identifier

        Returns:
            Dictionary with student behavior summary
        """
        node_id = f"student_{student_id}"
        if node_id not in self.graph:
            return {}

        summary = {
            "student_id": student_id,
            "locations_visited": [],
            "activities_performed": [],
            "mental_health_states": [],
            "demographics": self.graph.nodes[node_id].get("demographics", {})
        }

        # Get visited locations
        for target in self.graph.successors(node_id):
            for key, data in self.graph[node_id][target].items():
                if data.get("type") == RelationshipType.VISITED.value:
                    location_data = self.graph.nodes[target]
                    summary["locations_visited"].append({
                        "location_id": target,
                        "semantic_label": location_data.get("semantic_label"),
                        "timestamp": data.get("timestamp")
                    })
                elif data.get("type") == RelationshipType.PERFORMED.value:
                    activity_data = self.graph.nodes[target]
                    summary["activities_performed"].append({
                        "activity_id": target,
                        "activity_type": activity_data.get("activity_type"),
                        "timestamp": data.get("timestamp")
                    })
                elif data.get("type") == RelationshipType.EXPERIENCED.value:
                    mh_data = self.graph.nodes[target]
                    summary["mental_health_states"].append({
                        "state_id": target,
                        "phq4_score": mh_data.get("phq4_score"),
                        "anxiety_score": mh_data.get("anxiety_score"),
                        "depression_score": mh_data.get("depression_score"),
                        "timestamp": data.get("timestamp")
                    })

        return summary
