# arXiv Research Paper Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Analysis-green.svg)](https://networkx.org)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive_UI-red.svg)](https://gradio.app)

## Project Overview

An advanced **AI-powered research paper recommendation system** that analyzes 12,760+ arXiv papers across 80+ research topics using state-of-the-art machine learning techniques. The system combines **semantic similarity**, **graph neural networks**, and **knowledge graphs** to provide intelligent paper recommendations with interactive visualization.

### Key Features
- **Intelligent Recommendations**: Query-based paper suggestions using Universal Sentence Encoder
- **Knowledge Graph**: Complex network analysis with semantic similarity edges and co-author collaboration networks
- **Trend Analysis**: Real-time visualization of research topic evolution over time
- **Interactive Dashboard**: Web-based UI for exploration and visualization
- **Graph Neural Networks**: Advanced GCN implementation for enhanced recommendations
- **A/B Testing Framework**: Performance comparison and optimization metrics

---

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   Processing     │───▶│   ML Pipeline   │
│   (arXiv API)   │    │   & Cleaning     │    │   (USE + GCN)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  Knowledge       │    │   Evaluation    │
                    │  Graph           │    │   & A/B Tests   │
                    │  (NetworkX)      │    │                 │
                    └──────────────────┘    └─────────────────┘
                                │                       
                                ▼                       
                    ┌──────────────────┐    ┌─────────────────┐
                    │   Interactive    │◀───│  Trend Analysis │
                    │   Dashboard      │    │   (Time-series) │
                    │   (Gradio)       │    │                 │
                    └──────────────────┘    └─────────────────┘
                                ▲                       ▲
                                │                       │
                                └───── Processed ──────┘
                                       Data Flow
```

---

## Dataset
([url](https://www.kaggle.com/datasets/renukaoladhri/arxivr/data))

## Technical Implementation

### 1. **Data Collection & Processing**
```python
# Automated data collection from arXiv API
- 12,760 research papers across 80+ topics
- Real-time paper fetching with rate limiting
- Comprehensive metadata extraction
- Text preprocessing and cleaning
```

### 2. **Machine Learning Pipeline**
- **Universal Sentence Encoder (USE)**: 512-dimensional semantic embeddings
- **Graph Convolutional Networks (GCN)**: Enhanced relationship modeling
- **Cosine Similarity**: Efficient similarity computation
- **NetworkX**: Complex graph analysis and centrality measures

### 3. **Knowledge Graph Construction**
```python
Graph Components:
├── Paper Nodes (12,760)
├── Co-author Collaboration Edges
├── Category-based Connections  
└── Semantic Similarity Links (Top-K between papers)
```

### 4. **Advanced Features**
- **Multi-modal Recommendations**: Combining semantic + graph signals
- **Trend Prediction**: Time-series analysis of research topics  
- **Interactive Visualization**: Dynamic graph exploration
- **Performance Metrics**: Precision@K, nDCG@K evaluation

---

## Key Metrics & Results

| Metric | Value |
|--------|--------|
| **Papers Processed** | 12,760+ |
| **Research Topics** | 80+ |
| **Embedding Dimension** | 512 |
| **Graph Nodes** | 12,760+ |
| **Precision@5** | 0.80 |
| **nDCG@5** | 0.84 |
| **Response Time** | <2 seconds |

---

## Technology Stack

### **Core Technologies**
- **Python 3.8+**: Primary development language
- **TensorFlow 2.0+**: Deep learning framework
- **TensorFlow Hub**: Pre-trained model integration
- **NetworkX**: Graph analysis and visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

### **Visualization & UI**
- **Gradio**: Interactive web interface
- **Matplotlib**: Statistical plotting
- **Plotly**: Advanced data visualization

---

## Core Features Breakdown

### **Intelligent Recommendation Engine**
- Query-based paper discovery using semantic embeddings
- Multi-factor scoring combining similarity + graph metrics
- Real-time recommendation generation (<2s response time)

### **Trend Analysis Dashboard**
- Interactive visualization of research topic evolution
- Time-series analysis of publication patterns  
- Category-wise trend prediction and forecasting

### **Knowledge Graph Visualization**
- Dynamic network exploration with 10,000+ nodes
- Author collaboration networks
- Research topic clustering and communities

### **A/B Testing Framework**
- Performance comparison between algorithms
- Statistical significance testing
- Continuous model improvement pipeline

---

## Business Impact & Applications

### **Academic Research**
- Accelerate literature discovery by 60%
- Identify emerging research trends
- Facilitate cross-disciplinary collaboration

### **Industry Applications**  
- R&D department knowledge management
- Patent landscape analysis
- Competitive intelligence gathering

### **Educational Use Cases**
- Student research guidance
- Curriculum development insights
- Academic advisor tools

---

## Advanced Technical Details

### **Graph Neural Network Architecture**
```python
class GraphConvolutionalNetwork(tf.keras.Model):
    def __init__(self, hidden_dim=128, output_dim=512):
        super().__init__()
        self.gcn1 = GraphConvolution(hidden_dim)
        self.gcn2 = GraphConvolution(output_dim)
        
    def call(self, features, adjacency):
        x = tf.nn.relu(self.gcn1(features, adjacency))
        return self.gcn2(x, adjacency)
```

### **Recommendation Pipeline**
```python
def recommend_papers(query, top_k=5):
    # 1. Encode query using Universal Sentence Encoder
    query_embedding = encoder([query])
    
    # 2. Compute similarity with all papers  
    similarities = cosine_similarity(query_embedding, paper_embeddings)
    
    # 3. Apply graph-based boosting
    graph_scores = compute_centrality_scores(knowledge_graph)
    final_scores = similarities + α * graph_scores
    
    # 4. Return top-K recommendations
    return get_top_k_papers(final_scores, k=top_k)
```

---

## Performance Evaluation

### **Evaluation Metrics**
- **Precision@K**: Relevance of top-K recommendations
- **nDCG@K**: Normalized Discounted Cumulative Gain
- **Response Time**: System latency measurement
- **Graph Centrality**: Network importance metrics

### **A/B Test Results**
| Algorithm | Precision@5 | nDCG@5 | Avg Response Time |
|-----------|-------------|--------|-------------------|
| Baseline (Cosine) | 0.75 | 0.78 | 1.2s |
| **Graph-Enhanced** | **0.80** | **0.84** | **1.8s** |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
