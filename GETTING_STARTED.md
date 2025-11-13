# 🚀 Getting Started with Multi-Agent Mental Health Analysis

This guide will walk you through setting up and running the multi-agent system in 15 minutes.

## ⚡ Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull models
ollama pull llama3.1:70b
ollama pull nomic-embed-text

# 3. Configure Kaggle credentials
cp .env.example .env
# Edit .env with your KAGGLE_USERNAME and KAGGLE_KEY

# 4. Download data and build knowledge graph
python src/data_pipeline.py
python main.py --mode build-graph

# 5. Run demo
python main.py --mode demo

# Or query directly
python main.py --mode query --question "What locations correlate with better mental health?"
```

## 📝 Detailed Setup

### Step 1: Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Install all required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Ollama Setup

1. **Install Ollama**:
   - Visit [https://ollama.ai/](https://ollama.ai/)
   - Download and install for your OS

2. **Start Ollama**:
```bash
ollama serve
```

3. **Pull Required Models**:
```bash
# Large language model for general queries (16GB)
ollama pull llama3.1:70b

# Health-specific model for mental health analysis (16GB)
ollama pull meditron:70b

# Embedding model for semantic search (274MB)
ollama pull nomic-embed-text

# Optional: If you have limited resources, use smaller models
ollama pull llama3.1:8b  # Only 4.7GB
```

### Step 3: Kaggle API Setup

1. **Get Kaggle API Credentials**:
   - Go to [https://www.kaggle.com/](https://www.kaggle.com/)
   - Click on your profile → Account → API → Create New API Token
   - This downloads `kaggle.json`

2. **Configure Credentials**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your credentials:
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key

# Or place kaggle.json in:
# Linux/Mac: ~/.kaggle/kaggle.json
# Windows: C:\Users\<username>\.kaggle\kaggle.json
```

### Step 4: Download and Process Data

**Option A: Automated Pipeline (Recommended)**
```bash
python src/data_pipeline.py
```

This will:
- Download the StudentLife dataset from Kaggle
- Perform exploratory data analysis
- Clean and validate data
- Split into 6 agent-specific buckets
- Generate data quality report

**Option B: Manual Download**
1. Visit [https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset](https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset)
2. Download and extract to `data/raw/`
3. Run: `python src/data_pipeline.py`

### Step 5: Build Knowledge Graph

Create the knowledge graph from processed data:

```bash
python main.py --mode build-graph
```

This creates:
- **Student Nodes**: One per student with demographics
- **Location Nodes**: Semantic locations (gym, library, dorm, etc.)
- **Activity Nodes**: Physical activities and behaviors
- **Mental Health State Nodes**: PHQ4 assessments over time
- **Relationships**: VISITED, PERFORMED, EXPERIENCED, etc.

Expected output:
```
Knowledge Graph Statistics:
  num_nodes: 10,234
  num_edges: 52,441
  density: 0.001
  num_components: 1
```

### Step 6: Run Your First Query

Try the demo mode to see pre-configured queries:

```bash
python main.py --mode demo
```

Or ask your own question:

```bash
python main.py --mode query --question "What locations are associated with better mental health?"
```

### Step 7: Explore with Jupyter

Open the quick start notebook:

```bash
jupyter notebook notebooks/01_quick_start.ipynb
```

## 🎯 Example Workflows

### Workflow 1: Location-Mental Health Correlation

```bash
# Query spatial patterns
python main.py --mode query --question "Which locations do students with high PHQ4 scores visit most?"

# Compare with low PHQ4 students
python main.py --mode query --question "How do location patterns differ between students with good vs poor mental health?"
```

### Workflow 2: Temporal Analysis

```bash
# Analyze trends over time
python main.py --mode query --question "How do mental health scores change throughout the academic year?"

# Exam period impact
python main.py --mode query --question "How do PHQ4 scores change during exam periods vs regular weeks?"
```

### Workflow 3: Behavioral Predictions

```bash
# Identify risk factors
python main.py --mode query --question "What behavioral patterns predict mental health decline?"

# Generate predictions
python main.py --mode query --question "Predict next month's PHQ4 scores based on current sleep and activity patterns"
```

## 🔍 Verifying Installation

Check that everything is working:

```bash
# 1. Check system status
python main.py --mode status

# Expected output should show:
# - Orchestrator initialized
# - 6 specialized agents loaded
# - Knowledge graph statistics
# - Vector store count

# 2. Test Ollama connection
curl http://localhost:11434/api/tags

# Should return list of installed models

# 3. Verify data files exist
ls data/agent_buckets/

# Should show 6 parquet files:
# spatial_data.parquet
# behavioral_data.parquet
# mental_health_data.parquet
# temporal_data.parquet
# social_data.parquet
# demographic_data.parquet
```

## ❓ Common Issues

### Ollama Not Connecting

**Problem**: `Connection refused to localhost:11434`

**Solution**:
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### Out of Memory

**Problem**: System crashes when loading 70B models

**Solution**: Use smaller models
```bash
# Edit config/config.yaml
# Change:
llm:
  models:
    primary: "llama3.1:8b"  # Instead of 70b
    health_specific: "llama3.1:8b"  # Instead of meditron:70b
```

### Dataset Download Fails

**Problem**: Kaggle API authentication error

**Solution**:
```bash
# Verify credentials
cat ~/.kaggle/kaggle.json

# Ensure it has:
# {"username":"your_username","key":"your_api_key"}

# Test Kaggle CLI
kaggle datasets list

# Manual download if automated fails:
# Visit https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset
# Download → Extract to data/raw/
```

### Empty Knowledge Graph

**Problem**: Graph has 0 nodes after building

**Solution**:
```bash
# Check if data buckets exist
ls data/agent_buckets/

# Re-run data pipeline
python src/data_pipeline.py

# Rebuild graph
python main.py --mode build-graph
```

## 📚 Next Steps

1. **Read the Architecture**: Check `ARCHITECTURE.md` for system design details
2. **Explore Notebooks**: Open `notebooks/01_quick_start.ipynb`
3. **Review Config**: Customize `config/config.yaml` for your needs
4. **Try Advanced Queries**: Combine multiple agents for complex analysis
5. **Contribute**: Add new agents, analytics, or visualizations

## 💡 Tips for Best Results

1. **Start Small**: Use demo mode first to understand the system
2. **Use Specific Queries**: More specific questions get better answers
3. **Leverage Multiple Agents**: Ask questions that require spatial + behavioral + mental health analysis
4. **Check Confidence Scores**: Higher confidence = more reliable answers
5. **Review Agent Metadata**: See which agents contributed to each answer

## 🎓 Learning Resources

- **Ollama Documentation**: [https://ollama.ai/](https://ollama.ai/)
- **StudentLife Dataset Paper**: Wang et al. (2014)
- **Knowledge Graphs**: NetworkX documentation
- **Vector Databases**: ChromaDB documentation
- **Multi-Agent Systems**: LangChain/LangGraph tutorials

## ✅ Success Checklist

- [ ] Virtual environment created and activated
- [ ] All packages installed from requirements.txt
- [ ] Ollama installed and models pulled
- [ ] Kaggle credentials configured
- [ ] Dataset downloaded successfully
- [ ] Data pipeline completed
- [ ] Knowledge graph built
- [ ] Demo queries run successfully
- [ ] Custom query tested
- [ ] System status shows all agents operational

## 🎉 You're Ready!

Once you've completed the checklist above, you're ready to use the multi-agent system for advanced college mental health analysis!

Try this complete workflow:
```bash
python main.py --mode query --question "Analyze the relationship between gym visits, sleep quality, and mental health scores. Identify patterns and make recommendations."
```

This will engage multiple agents and demonstrate the full power of the system!

---

**Need Help?** Check `README_MULTIAGENT.md` for full documentation or open an issue on GitHub.
