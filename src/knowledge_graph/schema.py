"""Knowledge graph schema definitions."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    STUDENT = "Student"
    LOCATION = "Location"
    ACTIVITY = "Activity"
    MENTAL_HEALTH_STATE = "MentalHealthState"
    TEMPORAL_EVENT = "TemporalEvent"
    DEMOGRAPHIC = "Demographic"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    VISITED = "VISITED"
    PERFORMED = "PERFORMED"
    EXPERIENCED = "EXPERIENCED"
    AT_LOCATION = "AT_LOCATION"
    DURING = "DURING"
    CORRELATES_WITH = "CORRELATES_WITH"
    PRECEDED = "PRECEDED"
    BELONGS_TO = "BELONGS_TO"


@dataclass
class Node:
    """Base node class."""
    id: str
    type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False


@dataclass
class StudentNode(Node):
    """Student node."""

    def __init__(
        self,
        student_id: str,
        cohort: Optional[str] = None,
        demographics: Optional[Dict[str, Any]] = None,
        enrollment_year: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            id=f"student_{student_id}",
            type=NodeType.STUDENT,
            properties={
                "student_id": student_id,
                "cohort": cohort,
                "demographics": demographics or {},
                "enrollment_year": enrollment_year,
                **kwargs
            }
        )


@dataclass
class LocationNode(Node):
    """Location node."""

    def __init__(
        self,
        location_id: str,
        semantic_label: Optional[str] = None,
        coordinates: Optional[tuple] = None,
        location_type: Optional[str] = None,
        visit_count: int = 0,
        **kwargs
    ):
        super().__init__(
            id=f"location_{location_id}",
            type=NodeType.LOCATION,
            properties={
                "location_id": location_id,
                "semantic_label": semantic_label,
                "coordinates": coordinates,
                "location_type": location_type,
                "visit_count": visit_count,
                **kwargs
            }
        )


@dataclass
class ActivityNode(Node):
    """Activity node."""

    def __init__(
        self,
        activity_id: str,
        activity_type: str,
        duration: Optional[float] = None,
        intensity: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        super().__init__(
            id=f"activity_{activity_id}",
            type=NodeType.ACTIVITY,
            properties={
                "activity_id": activity_id,
                "activity_type": activity_type,
                "duration": duration,
                "intensity": intensity,
                "timestamp": timestamp,
                **kwargs
            }
        )


@dataclass
class MentalHealthStateNode(Node):
    """Mental health state node."""

    def __init__(
        self,
        state_id: str,
        phq4_score: Optional[float] = None,
        anxiety_score: Optional[float] = None,
        depression_score: Optional[float] = None,
        self_esteem: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            id=f"mental_health_{state_id}",
            type=NodeType.MENTAL_HEALTH_STATE,
            properties={
                "state_id": state_id,
                "phq4_score": phq4_score,
                "anxiety_score": anxiety_score,
                "depression_score": depression_score,
                "self_esteem": self_esteem,
                "timestamp": timestamp,
                "context": context or {},
                **kwargs
            }
        )


@dataclass
class TemporalEventNode(Node):
    """Temporal event node."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        impact_category: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            id=f"event_{event_id}",
            type=NodeType.TEMPORAL_EVENT,
            properties={
                "event_id": event_id,
                "event_type": event_type,
                "start_date": start_date,
                "end_date": end_date,
                "impact_category": impact_category,
                **kwargs
            }
        )


@dataclass
class DemographicNode(Node):
    """Demographic group node."""

    def __init__(
        self,
        demographic_id: str,
        cohort: Optional[str] = None,
        academic_year: Optional[int] = None,
        major: Optional[str] = None,
        background: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            id=f"demographic_{demographic_id}",
            type=NodeType.DEMOGRAPHIC,
            properties={
                "demographic_id": demographic_id,
                "cohort": cohort,
                "academic_year": academic_year,
                "major": major,
                "background": background or {},
                **kwargs
            }
        )


@dataclass
class Relationship:
    """Relationship between nodes."""
    source_id: str
    target_id: str
    rel_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    weight: float = 1.0

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.rel_type))

    def __eq__(self, other):
        if isinstance(other, Relationship):
            return (
                self.source_id == other.source_id
                and self.target_id == other.target_id
                and self.rel_type == other.rel_type
            )
        return False
