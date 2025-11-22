"""
Tax Fraud Detection - GNN Model Training with Real Datasets
Trains a Graph Convolutional Network on companies and invoices data
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from gnn_models.train_gnn import GNNFraudDetector, GNNTrainer

print("\n" + "=" * 80)
print("TRAINING GNN MODEL WITH REAL DATASETS")
print("=" * 80)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

print("\n📂 Loading processed data...")
data_dir = Path(__file__).parent / "data" / "processed"

companies = pd.read_csv(data_dir / "companies_processed.csv")
invoices = pd.read_csv(data_dir / "invoices_processed.csv")
features = pd.read_csv(data_dir / "company_features.csv")

print(f"✅ Companies: {len(companies)} records")
print(f"✅ Invoices: {len(invoices)} records")
print(f"✅ Features: {len(features)} vectors")

# ============================================================================
# 2. BUILD NETWORKX GRAPH
# ============================================================================

print("\n🔗 Building transaction network...")

# Create directed graph
G = nx.DiGraph()

# Add nodes with features
for idx, row in features.iterrows():
    G.add_node(
        row['company_id'],
        turnover=row['turnover'],
        sent_invoices=row['sent_invoices'],
        received_invoices=row['received_invoices'],
        fraud_label=row['is_fraud']
    )

# Add edges from invoices
invoices['date'] = pd.to_datetime(invoices['date'])
for idx, row in invoices.iterrows():
    if row['seller_id'] in G.nodes() and row['buyer_id'] in G.nodes():
        G.add_edge(
            row['seller_id'],
            row['buyer_id'],
            amount=row['amount'],
            itc_claimed=row['itc_claimed']
        )

print(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"✅ Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"✅ Network density: {nx.density(G):.6f}")

# ============================================================================
# 3. EXTRACT NODE FEATURES & LABELS
# ============================================================================

print("\n⚙️  Extracting features and labels...")

# Sort companies to ensure consistency
sorted_companies = sorted(G.nodes())
node_mapping = {company_id: idx for idx, company_id in enumerate(sorted_companies)}
reverse_mapping = {idx: company_id for company_id, idx in node_mapping.items()}

# Extract features (normalized)
feature_cols = ['turnover', 'sent_invoices', 'received_invoices']
X_features = []
y_labels = []

for company_id in sorted_companies:
    node_data = G.nodes[company_id]
    features_row = [
        float(node_data.get('turnover', 0)),
        float(node_data.get('sent_invoices', 0)),
        float(node_data.get('received_invoices', 0))
    ]
    X_features.append(features_row)
    y_labels.append(int(node_data.get('fraud_label', 0)))

X_features = np.array(X_features, dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int64)

# Normalize features (using log scale for turnover due to scale differences)
X_features[:, 0] = np.log1p(X_features[:, 0])  # Log transform turnover

scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

print(f"✅ Features shape: {X_features_scaled.shape}")
print(f"✅ Label distribution: {np.bincount(y_labels)}")

# ============================================================================
# 4. BUILD EDGE INDEX FOR PYTORCH GEOMETRIC
# ============================================================================

print("\n🔀 Building edge index for PyTorch Geometric...")

edge_list = []
for edge in G.edges():
    src_idx = node_mapping[edge[0]]
    dst_idx = node_mapping[edge[1]]
    edge_list.append([src_idx, dst_idx])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"✅ Edge index shape: {edge_index.shape}")

# ============================================================================
# 5. CREATE PYTORCH GEOMETRIC GRAPH
# ============================================================================

print("\n🔧 Creating PyTorch Geometric graph...")

x = torch.tensor(X_features_scaled, dtype=torch.float)
y = torch.tensor(y_labels, dtype=torch.long)

graph_data = Data(x=x, edge_index=edge_index, y=y)

print(f"✅ Graph data created:")
print(f"   Nodes: {graph_data.num_nodes}")
print(f"   Edges: {graph_data.num_edges}")
print(f"   Features per node: {graph_data.num_node_features}")

# Save graph data
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

# Ensure graphs directory exists before saving
graphs_dir = data_dir / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

torch.save(graph_data, graphs_dir / "graph_data.pt")
with open(graphs_dir / "node_mappings.pkl", 'wb') as f:
    pickle.dump(node_mapping, f)
# Write NetworkX graph using the readwrite gpickle writer (compatibility across NX versions)
try:
    from networkx.readwrite import gpickle as nx_gpickle
    nx_gpickle.write_gpickle(G, str(graphs_dir / "networkx_graph.gpickle"))
except Exception:
    # Fallback: pickle the NetworkX graph directly
    with open(graphs_dir / "networkx_graph.pkl", "wb") as f:
        pickle.dump(G, f)

print(f"✅ Graph data saved to {graphs_dir}")

# ============================================================================
# 6. TRAIN GNN MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🤖 TRAINING GNN MODEL")
print("=" * 80)

# Initialize trainer with correct data and models paths
trainer = GNNTrainer(data_path=str(data_dir), models_path=str(models_dir), model_type="gcn")

# Avoid torch.load inside the trainer (safe-globals issues with torch_geometric classes).
# Instead, assign the already-created graph_data to the trainer and run the pipeline steps manually.
trainer.data = graph_data.to(trainer.device)

# Create train/val/test split
trainer.create_train_val_test_split()

# Build model based on data features
in_channels = trainer.data.x.shape[1]
trainer.build_model(in_channels=in_channels, hidden_channels=64, out_channels=2)

# Train model
trainer.train_model(epochs=100, lr=0.001, weight_decay=5e-4)

# After training, trainer.model contains the trained model (best weights were saved and reloaded inside train_model)
trained_model = trainer.model
trained_model.eval()

with torch.no_grad():
    logits = trained_model(graph_data.x.to(trainer.device), graph_data.edge_index.to(trainer.device))
    predictions = torch.softmax(logits, dim=1)
    pred_labels = torch.argmax(logits, dim=1)
    fraud_proba = predictions[:, 1].cpu().numpy()

# ============================================================================
# 7. EVALUATION & RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)

# Get predictions on full dataset
model = trainer.model
model.eval()

with torch.no_grad():
    logits = model(graph_data.x, graph_data.edge_index)
    predictions = torch.softmax(logits, dim=1)
    pred_labels = torch.argmax(logits, dim=1)
    fraud_proba = predictions[:, 1].cpu().numpy()

# Create results dataframe
results_df = pd.DataFrame({
    'company_id': sorted_companies,
    'true_fraud': y_labels,
    'predicted_fraud': pred_labels.cpu().numpy(),
    'fraud_probability': fraud_proba
})

results_df.to_csv(data_dir / "model_predictions.csv", index=False)

# Calculate metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

accuracy = accuracy_score(y_labels, pred_labels.cpu().numpy())
precision = precision_score(y_labels, pred_labels.cpu().numpy(), zero_division=0)
recall = recall_score(y_labels, pred_labels.cpu().numpy(), zero_division=0)
f1 = f1_score(y_labels, pred_labels.cpu().numpy(), zero_division=0)
auc = roc_auc_score(y_labels, fraud_proba) if len(np.unique(y_labels)) > 1 else 0

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc:.4f}")

print(f"\n🎯 Confusion Matrix:")
tn, fp, fn, tp = confusion_matrix(y_labels, pred_labels.cpu().numpy()).ravel()
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

print(f"\n📋 Classification Report:")
print(classification_report(y_labels, pred_labels.cpu().numpy(), target_names=['Non-Fraud', 'Fraud']))

# ============================================================================
# 8. FRAUD DETECTION RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("🚨 FRAUD DETECTION RESULTS")
print("=" * 80)

fraud_detected = results_df[results_df['predicted_fraud'] == 1]
print(f"\n✅ Total companies flagged as fraud: {len(fraud_detected)}")
print(f"\n📋 High-Risk Companies (Fraud Probability > 0.7):")
high_risk = fraud_detected[fraud_detected['fraud_probability'] > 0.7].sort_values('fraud_probability', ascending=False)
print(high_risk[['company_id', 'true_fraud', 'fraud_probability']].head(10).to_string(index=False))

# ============================================================================
# 9. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"   Model: Graph Convolutional Network (GCN)")
print(f"   Nodes: {graph_data.num_nodes} companies")
print(f"   Edges: {graph_data.num_edges} transactions")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   F1-Score: {f1:.4f}")
print(f"   AUC-ROC: {auc:.4f}")

print(f"\n📁 Saved Files:")
print(f"   ✅ Model: {models_dir / 'best_model.pt'}")
print(f"   ✅ Graph: {data_dir / 'graphs' / 'graph_data.pt'}")
print(f"   ✅ Predictions: {data_dir / 'model_predictions.csv'}")
print(f"   ✅ Features: {data_dir / 'company_features.csv'}")

print(f"\n🚀 Next Steps:")
print(f"   1. Run Flask app: python app.py")
print(f"   2. Access dashboard at http://localhost:5000")
print(f"   3. View fraud predictions and analytics")

print("\n" + "=" * 80 + "\n")
