# Converting the uploaded invoice CSV into a GNN-ready node-classification dataset
# This cell will:
# 1. Load the CSV at /mnt/data/synthetic_data_11k_sf.csv
# 2. Build unique company nodes and aggregate node-level features
# 3. Build edge list with edge features (encoded)
# 4. Compute simple graph structural features using networkx
# 5. Create a PyTorch Geometric Data object and save .pt files + mappings + CSVs
# 6. Print a summary and sample of generated files

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
import networkx as nx

# Try import torch_geometric Data; if not available, we'll still save tensors
try:
    from torch_geometric.data import Data
    pyg_available = True
except Exception as e:
    Data = None
    pyg_available = False
    print("torch_geometric not available in this environment — will still save tensors and CSVs.")

# Paths
CSV_PATH = Path("/mnt/data/synthetic_data_11k_sf.csv")
OUT_DIR = Path("/mnt/data/gnn_ready_output")
OUT_DIR.mkdir(exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH, dtype=str)  # read as strings first to preserve formatting
# Convert numeric columns where possible
num_cols = ['Quantity','Rate_per_Unit','Total_Amount_ex_GST','CGST','SGST_UTGST','IGST',
            'Taxable_Value','Total_Invoice_Value_with_GST','ITC_Claimed','Amount']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Normalize date columns
date_cols = []
if 'Invoice_Date' in df.columns:
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'], errors='coerce')
    date_cols.append('Invoice_Date')
if 'Date' in df.columns and 'Date' not in date_cols:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    date_cols.append('Date')

# Fill missing Suspicious_Pattern_Flags with empty string
if 'Suspicious_Pattern_Flags' in df.columns:
    df['Suspicious_Pattern_Flags'] = df['Suspicious_Pattern_Flags'].fillna('')

# Standardize Fraud_Label to numeric 0/1
if 'Fraud_Label' in df.columns:
    df['Fraud_Label'] = pd.to_numeric(df['Fraud_Label'], errors='coerce').fillna(0).astype(int)
else:
    df['Fraud_Label'] = 0

# Build unique companies from Supplier_CIN and Buyer_CIN
supplier_col = 'Supplier_CIN'
buyer_col = 'Buyer_CIN'
nodes = pd.Index(df[supplier_col].fillna('NA_SUP').tolist() + df[buyer_col].fillna('NA_BUY').tolist())
unique_companies = pd.Index(nodes.unique())
node_mapping = {cin: idx for idx, cin in enumerate(unique_companies)}
inv_node_count = len(unique_companies)

# Aggregate node-level features
# Initialize dataframe for nodes
nodes_df = pd.DataFrame({
    'company_cin': list(unique_companies),
    'company_gid': [node_mapping[c] for c in unique_companies]
})

# Prepare helper columns in invoices for grouping
df['supplier_cin'] = df[supplier_col].fillna('UNKNOWN_SUP')
df['buyer_cin'] = df[buyer_col].fillna('UNKNOWN_BUY')
df['amount_num'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(df['Total_Amount_ex_GST'].fillna(0))
df['itc_num'] = pd.to_numeric(df['ITC_Claimed'], errors='coerce').fillna(0)
df['gst_total'] = pd.to_numeric(df.get('CGST',0), errors='coerce').fillna(0) + pd.to_numeric(df.get('SGST_UTGST',0), errors='coerce').fillna(0) + pd.to_numeric(df.get('IGST',0), errors='coerce').fillna(0)
df['is_suspicious_flag'] = df['Suspicious_Pattern_Flags'].astype(str).str.strip().ne('')

# Aggregations for sent (as supplier)
sent_agg = df.groupby('supplier_cin').agg(
    num_invoices_sent = ('Invoice_Number','count'),
    total_sent_amount = ('amount_num','sum'),
    avg_sent_amount = ('amount_num','mean'),
    total_itc_sent = ('itc_num','sum'),
    num_suspicious_sent = ('is_suspicious_flag','sum'),
    num_fraud_invoices_sent = ('Fraud_Label','sum'),
    unique_buyers = ('buyer_cin', lambda x: x.nunique()),
).reset_index().rename(columns={'supplier_cin':'company_cin'})

# Aggregations for received (as buyer)
recv_agg = df.groupby('buyer_cin').agg(
    num_invoices_received = ('Invoice_Number','count'),
    total_received_amount = ('amount_num','sum'),
    avg_received_amount = ('amount_num','mean'),
    total_itc_received = ('itc_num','sum'),
    num_suspicious_received = ('is_suspicious_flag','sum'),
    num_fraud_invoices_received = ('Fraud_Label','sum'),
    unique_suppliers = ('supplier_cin', lambda x: x.nunique()),
).reset_index().rename(columns={'buyer_cin':'company_cin'})

# Merge into nodes_df
nodes_df = nodes_df.merge(sent_agg, on='company_cin', how='left')
nodes_df = nodes_df.merge(recv_agg, on='company_cin', how='left')

# Fillna numeric zeros
num_fill_cols = ['num_invoices_sent','total_sent_amount','avg_sent_amount','total_itc_sent','num_suspicious_sent','num_fraud_invoices_sent','unique_buyers',
                 'num_invoices_received','total_received_amount','avg_received_amount','total_itc_received','num_suspicious_received','num_fraud_invoices_received','unique_suppliers']
for c in num_fill_cols:
    if c in nodes_df.columns:
        nodes_df[c] = nodes_df[c].fillna(0)

# Derived features
nodes_df['total_invoices'] = nodes_df['num_invoices_sent'] + nodes_df['num_invoices_received']
nodes_df['total_amount_all'] = nodes_df['total_sent_amount'] + nodes_df['total_received_amount']
nodes_df['avg_invoice_amount_all'] = nodes_df[['avg_sent_amount','avg_received_amount']].replace(0,np.nan).mean(axis=1).fillna(0)
nodes_df['total_itc_all'] = nodes_df['total_itc_sent'] + nodes_df['total_itc_received']
nodes_df['num_fraud_invoices'] = nodes_df['num_fraud_invoices_sent'] + nodes_df['num_fraud_invoices_received']
nodes_df['num_suspicious_flags'] = nodes_df['num_suspicious_sent'] + nodes_df['num_suspicious_received']
nodes_df['unique_partners'] = nodes_df['unique_buyers'].fillna(0) + nodes_df['unique_suppliers'].fillna(0)
nodes_df['itc_ratio'] = np.where(nodes_df['total_amount_all']>0, nodes_df['total_itc_all']/nodes_df['total_amount_all'], 0)

# Basic company metadata from first occurrence in df (supplier or buyer)
meta = {}
for idx, row in df.iterrows():
    sup = row['supplier_cin']
    buy = row['buyer_cin']
    # supplier metadata
    if sup not in meta:
        meta[sup] = {
            'company_name': row.get('Supplier_Name',''),
            'state': row.get('Supplier_State',''),
            'address': row.get('Supplier_Address','')
        }
    # buyer metadata
    if buy not in meta:
        meta[buy] = {
            'company_name': row.get('Buyer_Name',''),
            'state': row.get('Buyer_State',''),
            'address': row.get('Buyer_Address','')
        }

nodes_df['company_name'] = nodes_df['company_cin'].map(lambda x: meta.get(x,{}).get('company_name',''))
nodes_df['state'] = nodes_df['company_cin'].map(lambda x: meta.get(x,{}).get('state',''))
nodes_df['address'] = nodes_df['company_cin'].map(lambda x: meta.get(x,{}).get('address',''))

# Build edge list (from supplier -> buyer) with edge features
edges_df = df.copy()
edges_df['source'] = edges_df['supplier_cin'].map(node_mapping)
edges_df['target'] = edges_df['buyer_cin'].map(node_mapping)

# Encode Payment_Type and HSN_Code as categories (simple label encoding)
for col in ['Payment_Type','HSN_Code']:
    if col in edges_df.columns:
        edges_df[col] = edges_df[col].fillna('UNKNOWN').astype(str)
        edges_df[col + '_enc'] = edges_df[col].astype('category').cat.codes
    else:
        edges_df[col + '_enc'] = 0

# Edge features we will keep
edge_feature_cols = ['amount_num','taxable_value','itc_num','gst_total','Payment_Type_enc','HSN_Code_enc','is_suspicious_flag','Fraud_Label']
for c in edge_feature_cols:
    if c not in edges_df.columns:
        edges_df[c] = 0

# Some edges may have missing source/target mapping (drop those)
edges_df = edges_df[edges_df['source'].notna() & edges_df['target'].notna()]
edges_df = edges_df.astype({'source':int,'target':int})

# Build NetworkX directed graph for structural features
G = nx.DiGraph()
G.add_nodes_from(list(node_mapping.values()))
edge_tuples = list(zip(edges_df['source'].tolist(), edges_df['target'].tolist()))
G.add_edges_from(edge_tuples)

# Compute structural measures
pagerank = nx.pagerank(G) if G.number_of_nodes()>0 and G.number_of_edges()>0 else {n:0 for n in G.nodes()}
clustering = nx.clustering(G.to_undirected()) if G.number_of_nodes()>0 else {n:0 for n in G.nodes()}
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

# Map to nodes_df
nodes_df['pagerank'] = nodes_df['company_gid'].map(lambda x: pagerank.get(x,0))
nodes_df['clustering_coeff'] = nodes_df['company_gid'].map(lambda x: clustering.get(x,0))
nodes_df['in_degree'] = nodes_df['company_gid'].map(lambda x: in_deg.get(x,0))
nodes_df['out_degree'] = nodes_df['company_gid'].map(lambda x: out_deg.get(x,0))

# Node labels: mark company as fraudulent if it has >=1 fraud invoice (you can change threshold)
nodes_df['label'] = (nodes_df['num_fraud_invoices'] >= 1).astype(int)

# Prepare node feature matrix (select numeric features)
node_feature_columns = [
    'num_invoices_sent','num_invoices_received','total_sent_amount','total_received_amount',
    'avg_invoice_amount_all','total_amount_all','total_itc_all','itc_ratio','num_fraud_invoices',
    'num_suspicious_flags','unique_partners','pagerank','clustering_coeff','in_degree','out_degree'
]
# Ensure presence
node_feature_columns = [c for c in node_feature_columns if c in nodes_df.columns]
X = nodes_df[node_feature_columns].fillna(0).to_numpy(dtype=np.float32)

# Edge index and edge_attr tensors
edge_index = torch.tensor(np.vstack([edges_df['source'].to_numpy(), edges_df['target'].to_numpy()]), dtype=torch.long)
edge_attr = torch.tensor(edges_df[edge_feature_cols].fillna(0).to_numpy(dtype=np.float32), dtype=torch.float)

# Node label tensor
y = torch.tensor(nodes_df['label'].to_numpy(dtype=np.int64), dtype=torch.long)

# Save CSVs
nodes_df.to_csv(OUT_DIR / "company_nodes.csv", index=False)
edges_df.to_csv(OUT_DIR / "invoice_edges.csv", index=False)

# Save node mapping
with open(OUT_DIR / "node_mapping.pkl", "wb") as f:
    pickle.dump(node_mapping, f)

# Save tensors and PyG Data if available
torch.save({'x': X, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y}, OUT_DIR / "raw_tensors.pt")

if pyg_available:
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, y=y)
    torch.save(data, OUT_DIR / "graph_data.pt")
    print("Saved PyG graph_data.pt")
else:
    print("torch_geometric not available — saved raw_tensors.pt instead of graph_data.pt")

# Summary
print("=== SUMMARY ===")
print(f"Rows in original CSV: {len(df)}")
print(f"Unique companies (nodes): {len(unique_companies)}")
print(f"Edges (invoices): {len(edges_df)}")
print(f"Node feature matrix shape: {X.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge attr shape: {edge_attr.shape}")
print(f"Node labels distribution: {np.bincount(nodes_df['label'].astype(int))}")

# Show sample of nodes and edges
nodes_df_sample = nodes_df.head(5)
edges_df_sample = edges_df.head(5)[['source','target','Invoice_Number','amount_num','itc_num','gst_total','is_suspicious_flag','Fraud_Label']]
display(nodes_df_sample, edges_df_sample)

# List files created
import os
print("\nFiles saved to:", OUT_DIR)
for p in sorted(os.listdir(OUT_DIR)):
    print(" -", p)