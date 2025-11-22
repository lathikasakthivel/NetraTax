# Tax Fraud Detection Using Graph Neural Networks 🚀

## 📋 Project Overview

This project applies **Graph Neural Networks (GNNs)** to detect fraudulent tax patterns in Indian GST data by analyzing invoice transaction networks. The system identifies shell company networks and suspicious transaction chains that indicate tax fraud.

### 🎯 Key Objectives

1. **Build a transaction network** from invoice data (companies as nodes, invoices as edges)
2. **Extract network features** that capture fraudulent patterns
3. **Train a GNN model** to classify companies as fraudulent or legitimate
4. **Deploy an interactive dashboard** for tax auditors to identify high-risk companies
5. **Provide REST API** for integration with existing systems
6. **Integrate AI Chatbot** for conversational fraud analysis

---

## 🏗️ Project Structure

```
tax-fraud-gnn/
├── data/
│   ├── raw/
│   │   ├── companies.csv         # Company records (ID, turnover, location, is_fraud)
│   │   └── invoices.csv          # Invoice records (seller, buyer, amount, ITC)
│   └── processed/                # Cleaned & engineered data
│       ├── companies_processed.csv
│       ├── invoices_processed.csv
│       └── graphs/
│           ├── graph_data.pt                # PyTorch Geometric graph object
│           ├── networkx_graph.gpickle       # NetworkX format for analysis
│           └── node_mappings.pkl            # Company ID to node index mapping
├── models/
│   ├── best_model.pt             # Trained GNN model weights
│   ├── fraud_detector_model.pt    # Final model checkpoint
│   ├── model_metadata.json        # Model configuration
│   └── results.json               # Test metrics and confusion matrix
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── data_processing/
│   │   ├── clean_data.py         # Data cleaning and feature engineering
│   │   └── generate_sample_data.py # Generate synthetic test data
│   ├── graph_construction/
│   │   └── build_graph.py        # Build NetworkX & PyG graphs
│   ├── gnn_models/
│   │   └── train_gnn.py          # GNN training and evaluation
│   └── api/
│       └── app.py                # Flask REST API
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── chatbot.py                    # AI Chatbot for conversational analysis
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.9+
- Windows/Linux/macOS
- 4GB+ RAM (8GB recommended for GPU support)

### Step 1: Clone/Download Project

```bash
cd c:\BIG HACK
# Project is at: tax-fraud-gnn/
```

### Step 2: Create Virtual Environment (Windows)

```bash
cd tax-fraud-gnn
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you get PyTorch Geometric errors, visit [pyg.org](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for OS-specific installation.

### Step 4: Install Chatbot Dependencies

```bash
pip install -r chatbot_requirements.txt
```

---

## 📊 Workflow & Execution

### Phase 1: Data Preparation

#### Generate Sample Data (if you don't have real data)

```bash
cd src/data_processing
python generate_sample_data.py
```

**Output:** Creates `companies.csv` and `invoices.csv` in `data/raw/`

#### Clean & Process Data

```bash
python clean_data.py
```

**Output:** Creates processed datasets in `data/processed/`

Features engineered:
- `sent_invoice_count`: Number of invoices sent
- `received_invoice_count`: Number of invoices received
- `total_sent_amount`: Total value of sent invoices
- `total_received_amount`: Total value of received invoices

---

### Phase 2: Graph Construction

Build NetworkX and PyTorch Geometric graph representations:

```bash
cd ../graph_construction
python build_graph.py
```

**Output:** 
- `graph_data.pt` - PyTorch Geometric Data object
- `networkx_graph.gpickle` - NetworkX directed graph
- `node_mappings.pkl` - Mappings between company IDs and node indices

**Graph Structure:**
- **Nodes:** Companies (with attributes: turnover, location, is_fraud)
- **Edges:** Invoices (seller → buyer, weighted by amount)

---

### Phase 3: Model Training

Train the GNN for fraud classification:

```bash
cd ../gnn_models
python train_gnn.py
```

**Configuration:**
- Model: Graph Convolutional Network (GCN)
- Hidden layers: 64 neurons
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=20

**Output:**
- `best_model.pt` - Best model weights
- `fraud_detector_model.pt` - Final model
- `results.json` - Metrics (accuracy, precision, recall, F1, AUC-ROC)

---

### Phase 4: Dashboard & Visualization

Launch the interactive Streamlit dashboard:

```bash
cd ../../dashboard
streamlit run app.py
```

**Features:**
- 📊 Overview dashboard with fraud statistics
- 🔍 Detailed company analysis
- ⚠️ Risk scoring visualization
- 📈 Network insights and patterns

Opens at: `http://localhost:8501`

---

### Phase 5: REST API (Optional)

Deploy the Flask API for model serving:

```bash
cd ../src/api
python app.py
```

**Endpoints:**

- `GET /` - Health check
- `POST /api/predict` - Single company prediction
  ```json
  {"company_id": 123}
  ```
- `POST /api/batch_predict` - Multiple predictions
  ```json
  {"company_ids": [123, 456, 789]}
  ```
- `GET /api/company/<company_id>` - Company details
- `GET /api/stats` - Overall statistics

API runs on: `http://localhost:5000`

---

### Phase 6: AI Chatbot

Launch the AI-powered chatbot for conversational fraud analysis:

```bash
# Windows
start_chatbot.bat

# Linux/Mac
./start_chatbot.sh
```

**Features:**
- 🤖 Conversational interface for GST data analysis
- 📊 Real-time insights on companies and invoices
- ⚠️ Fraud pattern identification
- 💰 ITC claim analysis
- 📍 Location-based fraud detection

Chatbot runs on: `http://localhost:8501` (different port than dashboard)

---

## 📈 Expected Performance

### Sample Data (500 companies, 2000 invoices)

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~80% |
| Recall | ~75% |
| F1-Score | ~77% |
| AUC-ROC | ~0.88 |

*Actual results depend on data quality and model hyperparameters*

---

## 🔧 Configuration & Customization

### Model Hyperparameters

Edit in `src/gnn_models/train_gnn.py`:

```python
# In run_pipeline()
trainer.run_pipeline(
    epochs=100,          # Number of training epochs
    lr=0.001             # Learning rate
)
```

### GNN Architecture

Choose model type in `train_gnn.py`:

```python
trainer = GNNTrainer(model_type="gcn")  # or "graphsage"
```

### Data Processing

Adjust in `src/data_processing/clean_data.py`:
- Missing value handling strategy
- Feature engineering logic
- Data validation rules

---

## 📚 Input Data Format

### companies.csv

| Column | Type | Description |
|--------|------|-------------|
| company_id | int | Unique identifier |
| turnover | float | Annual turnover (₹) |
| location | string | Company location (state/city) |
| is_fraud | int | Ground truth label (0=normal, 1=fraud) |

### invoices.csv

| Column | Type | Description |
|--------|------|-------------|
| invoice_id | int | Unique invoice identifier |
| seller_id | int | Sending company ID |
| buyer_id | int | Receiving company ID |
| amount | float | Invoice amount (₹) |
| itc_claimed | float | Input Tax Credit claimed (₹) |

---

## 🚨 Fraud Detection Patterns

The GNN identifies fraudulent companies through:

1. **Network Topology:**
   - Unusual connection patterns
   - Hub-and-spoke structures (shell company networks)
   - Circular transaction chains

2. **Node Features:**
   - Disproportionate invoice frequency
   - Mismatched turnover vs. transaction volume
   - Location-based anomalies

3. **Edge Patterns:**
   - Unusually high ITC claims relative to invoice amounts
   - Rapid money cycling
   - Connectivity to known fraudsters

---

## 🔍 Exploratory Data Analysis

Jupyter notebook for data exploration:

```bash
cd notebooks
jupyter notebook eda.ipynb
```

**Includes:**
- Data distributions
- Fraud patterns visualization
- Network structure analysis
- Correlation studies

---

## ⚡ Performance Optimization

### For Large Datasets (>100K nodes)

1. **Use sampling:**
   ```python
   # In graph construction
   companies = companies.sample(frac=0.8)
   ```

2. **Enable GPU acceleration:**
   ```bash
   pip install torch-cuda  # For CUDA support
   ```

3. **Use mini-batch training:**
   ```python
   # Modify train_gnn.py to use DataLoader
   from torch_geometric.loader import DataLoader
   ```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` in activated venv |
| CUDA out of memory | Use CPU: set `CUDA_VISIBLE_DEVICES=""` or reduce batch size |
| Graph data not found | Run `build_graph.py` after cleaning data |
| Model weights not loading | Check `models/best_model.pt` exists |
| Port 8501 already in use | Kill process: `lsof -ti:8501 \| xargs kill -9` |

---

## 📚 References & Research

### Graph Neural Networks

- Kipf & Welling (2016): "Semi-Supervised Classification with GCNs"
- Hamilton et al. (2017): "GraphSAGE"
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

### Tax Fraud Detection

- Indian GST Fraud Detection Literature
- Network Analysis for Compliance
- Shell Company Detection Techniques

---

## 🤝 Contributing & Next Steps

**Potential Improvements:**

1. ✅ Implement GAT (Graph Attention Networks)
2. ✅ Add temporal dynamics (time-series invoice patterns)
3. ✅ Multi-hop fraud prediction
4. ✅ Ensemble methods combining multiple GNN models
5. ✅ Real-time batch prediction for new companies
6. ✅ Explainability (GNNExplainer for feature attribution)

---

## 📄 License & Disclaimer

This project is developed for **SIH 2024 Hackathon** and educational purposes.

**Disclaimer:** This system is a prototype. Real-world deployment requires:
- Validation on real GST data
- Compliance with Indian tax regulations
- Integration with authorized tax authorities
- Regular model retraining and validation

---

## 📞 Support

For issues or questions:

1. Check the Troubleshooting section
2. Review logs in your terminal
3. Verify data files are in correct directories
4. Ensure all dependencies are installed

---

**Last Updated:** November 2025  
**Status:** ✅ Production Ready  
**Maintainers:** SIH 2024 Team

---

🚀 **Happy Fraud Detection!**