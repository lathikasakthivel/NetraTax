"""
Train GNN Model from Uploaded CSV Files
Processes companies.csv and invoices.csv from uploads folder,
builds graph, trains model, and saves everything for dashboard
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
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from gnn_models.train_gnn import GNNFraudDetector, GNNTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("TRAINING GNN MODEL FROM UPLOADED CSV FILES")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA FROM UPLOADS FOLDER
# ============================================================================

print("\n[1/6] Loading CSV files from uploads folder...")
uploads_path = Path(__file__).parent / "data" / "uploads"

# Find the most recent upload folder
companies_file = None
invoices_file = None

if uploads_path.exists():
    upload_folders = sorted([d for d in uploads_path.iterdir() if d.is_dir()], reverse=True)
    for upload_folder in upload_folders:
        companies_csv = upload_folder / "companies.csv"
        invoices_csv = upload_folder / "invoices.csv"
        if companies_csv.exists() and companies_file is None:
            companies_file = companies_csv
        if invoices_csv.exists() and invoices_file is None:
            invoices_file = invoices_csv
        if companies_file and invoices_file:
            break

if companies_file is None or invoices_file is None:
    raise FileNotFoundError("Could not find companies.csv and invoices.csv in uploads folder")

print(f"Found companies.csv: {companies_file}")
print(f"Found invoices.csv: {invoices_file}")

# Load CSV files
companies = pd.read_csv(companies_file)
invoices = pd.read_csv(invoices_file)

print(f"Loaded {len(companies)} companies")
print(f"Loaded {len(invoices)} invoices")

# ============================================================================
# 2. CLEAN AND PREPARE DATA
# ============================================================================

print("\n[2/6] Cleaning and preparing data...")

# Clean companies data
if 'company_id' not in companies.columns:
    raise ValueError("companies.csv must have 'company_id' column")
if 'is_fraud' not in companies.columns:
    companies['is_fraud'] = 0  # Default to non-fraud if not present
if 'turnover' not in companies.columns:
    companies['turnover'] = 0.0
if 'location' not in companies.columns:
    companies['location'] = 'Unknown'

# Clean invoices data
if 'seller_id' not in invoices.columns or 'buyer_id' not in invoices.columns:
    raise ValueError("invoices.csv must have 'seller_id' and 'buyer_id' columns")
if 'amount' not in invoices.columns:
    invoices['amount'] = 0.0
if 'itc_claimed' not in invoices.columns:
    invoices['itc_claimed'] = 0

# Convert data types
companies['company_id'] = companies['company_id'].astype(str).str.strip()
invoices['seller_id'] = invoices['seller_id'].astype(str).str.strip()
invoices['buyer_id'] = invoices['buyer_id'].astype(str).str.strip()
companies['is_fraud'] = pd.to_numeric(companies['is_fraud'], errors='coerce').fillna(0).astype(int)
companies['turnover'] = pd.to_numeric(companies['turnover'], errors='coerce').fillna(0.0)
invoices['amount'] = pd.to_numeric(invoices['amount'], errors='coerce').fillna(0.0)
invoices['itc_claimed'] = pd.to_numeric(invoices['itc_claimed'], errors='coerce').fillna(0).astype(int)

# Remove self-loops
invoices = invoices[invoices['seller_id'] != invoices['buyer_id']]

print(f"After cleaning: {len(companies)} companies, {len(invoices)} invoices")

# ============================================================================
# 3. ENGINEER FEATURES
# ============================================================================

print("\n[3/6] Engineering features...")

# Calculate invoice statistics per company
sent_invoices = invoices.groupby('seller_id').size().reset_index(name='sent_invoices')
received_invoices = invoices.groupby('buyer_id').size().reset_index(name='received_invoices')
sent_amount = invoices.groupby('seller_id')['amount'].sum().reset_index(name='total_sent_amount')
received_amount = invoices.groupby('buyer_id')['amount'].sum().reset_index(name='total_received_amount')

# Merge with companies
companies = companies.merge(sent_invoices, left_on='company_id', right_on='seller_id', how='left')
companies = companies.merge(received_invoices, left_on='company_id', right_on='buyer_id', how='left')
companies = companies.merge(sent_amount, left_on='company_id', right_on='seller_id', how='left')
companies = companies.merge(received_amount, left_on='company_id', right_on='buyer_id', how='left')

# Fill NaN values
companies['sent_invoices'] = companies['sent_invoices'].fillna(0).astype(int)
companies['received_invoices'] = companies['received_invoices'].fillna(0).astype(int)
companies['total_sent_amount'] = companies['total_sent_amount'].fillna(0.0)
companies['total_received_amount'] = companies['total_received_amount'].fillna(0.0)

# Calculate invoice frequency
companies['invoice_frequency'] = companies['sent_invoices'] + companies['received_invoices']

print(f"Feature engineering complete. Companies with features: {len(companies)}")

# ============================================================================
# 4. BUILD NETWORKX GRAPH
# ============================================================================

print("\n[4/6] Building transaction network graph...")

# Create directed graph
G = nx.DiGraph()

# Add nodes with features
for idx, row in companies.iterrows():
    G.add_node(
        row['company_id'],
        turnover=float(row['turnover']),
        sent_invoices=int(row['sent_invoices']),
        received_invoices=int(row['received_invoices']),
        fraud_label=int(row['is_fraud']),
        location=str(row.get('location', 'Unknown'))
    )

# Add edges from invoices
for idx, row in invoices.iterrows():
    seller = str(row['seller_id']).strip()
    buyer = str(row['buyer_id']).strip()
    if seller in G.nodes() and buyer in G.nodes():
        G.add_edge(
            seller,
            buyer,
            amount=float(row['amount']),
            itc_claimed=int(row['itc_claimed'])
        )

print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"Network density: {nx.density(G):.6f}")

# ============================================================================
# 5. CONVERT TO PYTORCH GEOMETRIC FORMAT
# ============================================================================

print("\n[5/6] Converting to PyTorch Geometric format...")

# Sort companies to ensure consistency
sorted_companies = sorted(G.nodes())
node_mapping = {company_id: idx for idx, company_id in enumerate(sorted_companies)}
reverse_mapping = {idx: company_id for company_id, idx in node_mapping.items()}

# Extract features and labels
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

# Normalize features
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Build edge index
edge_list = []
for edge in G.edges():
    src_idx = node_mapping[edge[0]]
    dst_idx = node_mapping[edge[1]]
    edge_list.append([src_idx, dst_idx])

if len(edge_list) == 0:
    raise ValueError("No edges found in graph. Check invoice data.")

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Create PyTorch Geometric Data object
x = torch.tensor(X_features_scaled, dtype=torch.float)
y = torch.tensor(y_labels, dtype=torch.long)

graph_data = Data(x=x, edge_index=edge_index, y=y)

print(f"PyTorch Geometric graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
print(f"Features per node: {graph_data.num_node_features}")
print(f"Fraud labels: {np.bincount(y_labels)}")

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================

print("\n[6/6] Training GNN model...")

# Setup paths
data_path = Path(__file__).parent / "data" / "processed"
models_path = Path(__file__).parent / "models"
graphs_path = data_path / "graphs"

# Create directories
data_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)
graphs_path.mkdir(parents=True, exist_ok=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GNNFraudDetector(
    in_channels=3,
    hidden_channels=64,
    out_channels=2,
    model_type="gcn"
).to(device)

# Train model using a simpler approach
print("Setting up training...")

# Create train/val/test split
num_nodes = graph_data.num_nodes
y = graph_data.y

# Get indices of each class
fraud_idx = np.where(y.cpu().numpy() == 1)[0]
normal_idx = np.where(y.cpu().numpy() == 0)[0]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split each class
fraud_train = int(len(fraud_idx) * train_ratio)
fraud_val = int(len(fraud_idx) * val_ratio)

normal_train = int(len(normal_idx) * train_ratio)
normal_val = int(len(normal_idx) * val_ratio)

train_idx = torch.tensor(np.concatenate([fraud_idx[:fraud_train], normal_idx[:normal_train]]), dtype=torch.long, device=device)
val_idx = torch.tensor(np.concatenate([
    fraud_idx[fraud_train:fraud_train+fraud_val],
    normal_idx[normal_train:normal_train+normal_val]
]), dtype=torch.long, device=device)
test_idx = torch.tensor(np.concatenate([
    fraud_idx[fraud_train+fraud_val:],
    normal_idx[normal_train+normal_val:]
]), dtype=torch.long, device=device)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Move data to device
graph_data = graph_data.to(device)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Compute class weights
y_all = y.cpu().numpy()
unique, counts = np.unique(y_all, return_counts=True)
weights = np.ones((2,), dtype=np.float32)
if len(unique) == 2:
    for cls, cnt in zip(unique, counts):
        weights[int(cls)] = 1.0 / float(cnt)
    weights = weights / weights.sum() * len(weights)
weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

print("Starting training...")
epochs = 100
best_val_acc = 0
patience = 20
patience_counter = 0

for epoch in range(epochs):
    # Train
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out[train_idx], graph_data.y[train_idx])
    loss.backward()
    optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        val_loss = criterion(out[val_idx], graph_data.y[val_idx])
        pred = out[val_idx].argmax(dim=1)
        val_acc = (pred == graph_data.y[val_idx]).float().mean().item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), models_path / "best_model.pt")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load(models_path / "best_model.pt"))
print("Training complete!")

# ============================================================================
# 7. SAVE EVERYTHING
# ============================================================================

print("\n[Saving] Saving processed data, graph, and model...")

# Save processed data
companies.to_csv(data_path / "companies_processed.csv", index=False)
invoices.to_csv(data_path / "invoices_processed.csv", index=False)
print(f"Saved processed data to {data_path}")

# Save graph
torch.save(graph_data, graphs_path / "graph_data.pt")
with open(graphs_path / "node_mappings.pkl", 'wb') as f:
    pickle.dump(node_mapping, f)

# Save NetworkX graph
try:
    from networkx.readwrite import gpickle as nx_gpickle
    nx_gpickle.write_gpickle(G, str(graphs_path / "networkx_graph.gpickle"))
except Exception:
    with open(graphs_path / "networkx_graph.pkl", "wb") as f:
        pickle.dump(G, f)

print(f"Saved graph data to {graphs_path}")

# Save model
torch.save(model.state_dict(), models_path / "best_model.pt")
print(f"Saved model to {models_path / 'best_model.pt'}")

# ============================================================================
# 8. GENERATE PREDICTIONS AND STATISTICS
# ============================================================================

print("\n[Finalizing] Generating predictions...")

model.eval()
with torch.no_grad():
    out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
    predictions = torch.softmax(out, dim=1)
    fraud_proba = predictions[:, 1].cpu().numpy()

# Add predictions to companies dataframe
companies['fraud_probability'] = fraud_proba
companies['predicted_fraud'] = (fraud_proba > 0.5).astype(int)

# Save predictions
companies.to_csv(data_path / "companies_processed.csv", index=False)

# Print statistics
print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)
print(f"Total Companies: {len(companies)}")
print(f"Total Invoices: {len(invoices)}")
print(f"Graph Nodes: {graph_data.num_nodes}")
print(f"Graph Edges: {graph_data.num_edges}")
print(f"Average Fraud Probability: {fraud_proba.mean():.4f}")
print(f"High-Risk Companies (prob > 0.5): {(fraud_proba > 0.5).sum()}")
print(f"High-Risk Companies (prob > 0.7): {(fraud_proba > 0.7).sum()}")
print("=" * 80)
print("\nModel and data are ready for dashboard!")
print("Restart your Flask server to see the results.")

