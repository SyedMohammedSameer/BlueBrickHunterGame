# Multi-Agent College Mental Health Analysis System
## System Architecture Documentation

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (Gradio)                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │  Query   │  KG Viz  │ Similar. │Predictive│Agent Debug   │  │
│  │Interface │          │  Search  │Dashboard │              │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Query Decomposition  • Agent Routing • Result Synthesis │  │
│  │  • Conflict Resolution  • Error Handling  • Caching        │  │
│  └───────────────────────────────────────────────────────────┘  │
└───┬───────┬───────┬───────┬───────┬───────┬───────────────────┘
    │       │       │       │       │       │
    ▼       ▼       ▼       ▼       ▼       ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────────┐
│Spatial ││Behavio-││Mental  ││Temporal││Social  ││Demographic │
│Agent   ││ral     ││Health  ││Agent   ││Agent   ││Agent       │
│        ││Agent   ││Agent   ││        ││        ││            │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└─────┬──────┘
    │         │         │         │         │           │
    └─────────┴─────────┴─────────┴─────────┴───────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA & KNOWLEDGE LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Knowledge   │  │   Vector     │  │   Agent Data         │  │
│  │    Graph     │  │   Store      │  │   Buckets            │  │
│  │  (NetworkX)  │  │ (ChromaDB)   │  │  (Parquet Files)     │  │
│  │              │  │              │  │                      │  │
│  │ • Students   │  │ • Behaviors  │  │ • spatial_data.pq    │  │
│  │ • Locations  │  │ • Contexts   │  │ • behavioral_data.pq │  │
│  │ • Activities │  │ • Narratives │  │ • mental_health.pq   │  │
│  │ • States     │  │ • Patterns   │  │ • temporal_data.pq   │  │
│  │ • Events     │  │              │  │ • social_data.pq     │  │
│  │              │  │              │  │ • demographic_data.pq│  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYTICS ENGINE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Causal     │  │  Predictive  │  │   Anomaly            │  │
│  │  Inference   │  │   Models     │  │   Detection          │  │
│  │   (DoWhy)    │  │(XGBoost/LSTM)│  │(Isolation Forest)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Clustering  │  │   Network    │  │   Statistical        │  │
│  │  (HDBSCAN)   │  │  Analysis    │  │   Tests              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM LAYER (Ollama)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Llama 70B   │  │  Meditron 70B│  │  Nomic Embeddings    │  │
│  │  (Primary)   │  │  (MH Focus)  │  │  (768-dim)           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🤖 Agent Architecture

### Agent Communication Protocol

```python
class AgentMessage:
    from_agent: str          # Source agent identifier
    to_agent: str            # Destination agent (or "orchestrator")
    query: str               # Natural language query
    data: Dict[str, Any]     # Structured data payload
    priority: int            # 1-5, where 5 is highest
    timestamp: datetime      # Message creation time
    context: Dict[str, Any]  # Shared context from previous messages
```

### Agent Base Class

All agents inherit from `BaseAgent` with:
- **Tools**: Query graph, query vectors, compute stats, call LLM
- **State**: Conversation history, cache, performance metrics
- **Methods**: `process_query()`, `use_tool()`, `communicate()`

### Agent Specializations

#### 1. SpatialAgent
**Data Bucket**: `spatial_data.parquet`
- GPS coordinates, location semantics, mobility patterns
- Location visit frequency, dwell times, movement patterns

**Capabilities**:
- Location clustering and semantic labeling
- Mobility pattern analysis (home-gym-study patterns)
- Geospatial correlation with mental health
- Commute analysis and travel behavior

**Example Queries**:
- "Which locations correlate with better mental health?"
- "Show mobility patterns during exam periods"
- "Find students with similar location visit patterns"

#### 2. BehavioralAgent
**Data Bucket**: `behavioral_data.parquet`
- Activity recognition, sleep patterns, phone usage
- Screen time, app usage, physical activity

**Capabilities**:
- Sleep quality analysis and pattern detection
- Activity level tracking and anomaly detection
- Phone usage correlation with mental health
- Behavioral routine identification

**Example Queries**:
- "How does sleep quality affect PHQ4 scores?"
- "Identify abnormal activity patterns"
- "Correlate screen time with mental health"

#### 3. MentalHealthAgent
**Data Bucket**: `mental_health_data.parquet`
- PHQ4 scores, self-esteem measures, anxiety/depression metrics
- Survey responses, clinical assessments

**Capabilities**:
- PHQ4 score prediction and trend analysis
- Risk stratification and early warning
- Mental health trajectory modeling
- Intervention recommendation

**Example Queries**:
- "Predict PHQ4 scores for next month"
- "Identify students at risk"
- "What factors predict mental health decline?"

#### 4. TemporalAgent
**Data Bucket**: `temporal_data.parquet`
- Time-series data, temporal events, longitudinal patterns
- Academic calendar, holidays, exam periods

**Capabilities**:
- Time-series forecasting and trend analysis
- Event-based pattern detection
- Longitudinal trajectory modeling
- Seasonal pattern identification

**Example Queries**:
- "How do exam periods affect mental health?"
- "Show temporal trends in sleep patterns"
- "Predict future behavioral patterns"

#### 5. SocialAgent
**Data Bucket**: `social_data.parquet`
- Social interactions, communication patterns
- Screen time, social app usage

**Capabilities**:
- Social network analysis
- Interaction pattern detection
- Social isolation identification
- Communication behavior analysis

**Example Queries**:
- "How do social interactions affect mental health?"
- "Identify socially isolated students"
- "Correlate communication patterns with PHQ4"

#### 6. DemographicAgent
**Data Bucket**: `demographic_data.parquet`
- Student demographics, cohort information
- Academic year, major, background

**Capabilities**:
- Cohort analysis and comparison
- Demographic-based pattern detection
- Equity and fairness analysis
- Subgroup identification

**Example Queries**:
- "Compare mental health across cohorts"
- "Identify demographic disparities"
- "Analyze trends by academic year"

#### 7. OrchestratorAgent
**No Data Bucket** (coordinates other agents)

**Capabilities**:
- Query decomposition into sub-queries
- Intelligent agent routing
- Response synthesis and conflict resolution
- Context management across agents
- Performance optimization and caching

**Workflow**:
```
1. Receive user query
2. Analyze query complexity and domain
3. Decompose into agent-specific sub-queries
4. Route to appropriate agents (parallel when possible)
5. Collect and validate responses
6. Resolve conflicts and synthesize insights
7. Generate natural language response
8. Update cache and metrics
```

## 📊 Knowledge Graph Schema

### Node Types

```
Student {
  id: String
  cohort: String
  demographics: Dict
  enrollment_year: Int
}

Location {
  semantic_label: String  # gym, dorm, library, etc.
  coordinates: Tuple[Float, Float]
  type: String  # academic, residential, recreational
  visit_count: Int
}

Activity {
  type: String  # walking, running, stationary, etc.
  duration: Float
  intensity: String  # low, medium, high
  timestamp: DateTime
}

MentalHealthState {
  phq4_score: Float
  anxiety_score: Float
  depression_score: Float
  self_esteem: Float
  timestamp: DateTime
  context: Dict
}

TemporalEvent {
  event_type: String  # exam, holiday, campus_event
  start_date: DateTime
  end_date: DateTime
  impact_category: String
}

Demographic {
  cohort: String
  academic_year: Int
  major: String
  background: Dict
}
```

### Relationship Types

```
(Student)-[:VISITED {timestamp, duration}]->(Location)
(Student)-[:PERFORMED {timestamp}]->(Activity)
(Student)-[:EXPERIENCED {timestamp}]->(MentalHealthState)
(Student)-[:BELONGS_TO]->(Demographic)
(Activity)-[:AT_LOCATION]->(Location)
(MentalHealthState)-[:DURING]->(TemporalEvent)
(Location)-[:CORRELATES_WITH {correlation, p_value}]->(MentalHealthState)
(Activity)-[:PRECEDED {time_delta}]->(Activity)
```

## 🎯 Data Flow

### Ingestion Pipeline

```
Kaggle Dataset
      ↓
Data Loading (Polars)
      ↓
Data Cleaning & Validation
      ↓
Feature Engineering
      ↓
Data Splitting (6 agent buckets)
      ↓
Knowledge Graph Construction
      ↓
Embedding Generation
      ↓
Vector Store Indexing
```

### Query Processing Pipeline

```
User Query
      ↓
Orchestrator (Query Analysis)
      ↓
Agent Routing Decision
      ↓
Parallel Agent Execution
   ↓           ↓           ↓
Spatial    Behavioral   Mental Health
Agent         Agent         Agent
   ↓           ↓           ↓
Tool Execution (Graph + Vector + Stats)
      ↓
Response Collection
      ↓
Conflict Resolution
      ↓
Synthesis (LLM-based)
      ↓
User Response
```

## 🔧 Technology Stack Details

### Core Technologies
- **Python 3.10+**: Main language
- **Polars**: High-performance data processing
- **NetworkX**: Knowledge graph backend
- **ChromaDB**: Vector store for embeddings
- **Ollama**: Local LLM inference
- **LangGraph**: Agent orchestration framework

### Machine Learning
- **PyTorch**: Deep learning models
- **XGBoost**: Gradient boosting for predictions
- **scikit-learn**: Classical ML algorithms
- **HDBSCAN**: Density-based clustering
- **DoWhy**: Causal inference

### Visualization
- **Gradio**: Interactive web interface
- **PyVis**: Interactive knowledge graph visualization
- **Plotly**: Interactive plots and dashboards
- **Matplotlib/Seaborn**: Static visualizations

## 🚀 Performance Optimizations

### Caching Strategy
- **Query Cache**: LRU cache for frequent queries
- **Embedding Cache**: Precomputed embeddings for common patterns
- **Graph Cache**: Cached graph traversal results

### Parallel Processing
- **Agent Parallelism**: Independent agents run concurrently
- **Batch Processing**: Data processing in batches
- **GPU Acceleration**: PyTorch operations on GPU

### Scalability
- **Lazy Loading**: Load data on-demand
- **Streaming**: Process large datasets in chunks
- **Index Optimization**: Optimized graph and vector indices

## 📈 Evaluation Framework

### Metrics

**Retrieval Quality**:
- Precision@K, Recall@K, NDCG
- Graph query latency
- Vector search accuracy

**Prediction Quality**:
- RMSE, MAE, R² for regression
- Accuracy, F1, AUC-ROC for classification
- Calibration error

**Agent Performance**:
- Response relevance (human-evaluated)
- Query decomposition accuracy
- Inter-agent communication efficiency

**System Performance**:
- End-to-end latency
- Throughput (queries/second)
- Resource utilization

## 🔒 Privacy & Ethics

### Data Privacy
- No PII in logs or outputs
- Differential privacy for aggregate statistics
- Secure data storage and access control

### Ethical Considerations
- Bias detection in predictions
- Fairness across demographic groups
- Transparent decision-making
- Human oversight for sensitive predictions

## 📚 Project Structure

```
BlueBrickHunterGame/
├── config/
│   ├── config.yaml              # Main configuration
│   └── benchmark_queries.json   # Evaluation queries
├── data/
│   ├── raw/                     # Raw Kaggle data
│   ├── processed/               # Cleaned data
│   ├── agent_buckets/           # Agent-specific data
│   └── chromadb/                # Vector store
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Base agent class
│   │   ├── orchestrator.py      # Orchestrator agent
│   │   ├── spatial_agent.py     # Spatial agent
│   │   ├── behavioral_agent.py  # Behavioral agent
│   │   ├── mental_health_agent.py
│   │   ├── temporal_agent.py
│   │   ├── social_agent.py
│   │   └── demographic_agent.py
│   ├── knowledge_graph/
│   │   ├── graph_builder.py     # KG construction
│   │   ├── graph_query.py       # Query interface
│   │   └── schema.py            # Graph schema
│   ├── embeddings/
│   │   ├── embedding_generator.py
│   │   └── vector_store.py
│   ├── analytics/
│   │   ├── causal_inference.py
│   │   ├── predictive_models.py
│   │   ├── anomaly_detection.py
│   │   └── clustering.py
│   ├── interface/
│   │   ├── gradio_app.py        # Main UI
│   │   └── visualizations.py    # Viz components
│   └── utils/
│       ├── data_loader.py
│       ├── config_loader.py
│       ├── logger.py
│       └── metrics.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_kg_construction.ipynb
│   ├── 03_agent_testing.ipynb
│   └── 04_analytics.ipynb
├── tests/
│   ├── test_agents.py
│   ├── test_knowledge_graph.py
│   └── test_analytics.py
├── models/                      # Saved models
├── outputs/
│   ├── visualizations/          # Generated plots
│   └── reports/                 # Research reports
├── logs/                        # System logs
├── requirements.txt
├── .env.example
├── README.md
└── ARCHITECTURE.md              # This file
```

## 🎓 Research Applications

This system enables research in:
- **Digital Phenotyping**: Passive sensing of mental health
- **Behavioral Interventions**: Data-driven recommendations
- **Predictive Psychiatry**: Early warning systems
- **Causal Analysis**: Understanding mental health determinants
- **Precision Medicine**: Personalized mental health care

## 📖 References

- StudentLife Dataset (Dartmouth College)
- Digital Phenotyping Research
- Knowledge Graph Embeddings
- Multi-Agent Systems for Healthcare
- Causal Inference in Observational Studies
