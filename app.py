"""
Flask Web Application for Tax Fraud Detection
Replace Streamlit with professional HTML/CSS interface
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
import logging
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os
import networkx as nx
from collections import deque
import torch.nn as nn
from flask_cors import CORS
from torch_geometric.data import Data  # Add this import

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules with error handling
GNNFraudDetector = None
init_db = None
record_upload = None
list_uploads = None
encrypt_file = None

try:
    from src.gnn_models.train_gnn import GNNFraudDetector
    from src.db import init_db, record_upload, list_uploads
    from src.crypto import encrypt_file
except ImportError as e:
    logger.error(f"Import error: {e}")

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False
CORS(app)  # Enable CORS for React dev server

# Custom JSON encoder for numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        return super(NumpyEncoder, self).default(o)

# For Flask 2.0+, use json.default instead of json_encoder
app.json.default = lambda o: NumpyEncoder().default(o)

# Global variables for model and data
MODEL = None
GRAPH_DATA = None
DEVICE = None
COMPANIES = None
INVOICES = None
MAPPINGS = None
FRAUD_PROBA = None
NETWORKX_GRAPH = None  # Add NetworkX graph for easier manipulation


def load_model_and_data():
    """Load model and data on startup"""
    global MODEL, GRAPH_DATA, DEVICE, COMPANIES, INVOICES, MAPPINGS, FRAUD_PROBA, NETWORKX_GRAPH
    
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    uploads_path = Path(__file__).parent / "data" / "uploads"
    
    logger.info("Loading data...")
    
    # Try to load from uploads folder first (most recent data)
    companies_file = None
    invoices_file = None
    
    # Find the most recent upload folder
    if uploads_path.exists():
        upload_folders = sorted([d for d in uploads_path.iterdir() if d.is_dir()], reverse=True)
        for upload_folder in upload_folders:
            companies_csv = upload_folder / "companies.csv"
            invoices_csv = upload_folder / "invoices.csv"
            if companies_csv.exists() and companies_file is None:
                companies_file = companies_csv
                logger.info(f"Found companies.csv in uploads: {companies_csv}")
            if invoices_csv.exists() and invoices_file is None:
                invoices_file = invoices_csv
                logger.info(f"Found invoices.csv in uploads: {invoices_csv}")
            if companies_file and invoices_file:
                break
    
    # Fallback to processed data if not found in uploads
    if companies_file is None:
        companies_file = data_path / "companies_processed.csv"
        if not companies_file.exists():
            companies_file = data_path.parent / "raw" / "companies.csv"
        logger.info(f"Loading companies from: {companies_file}")
    
    if invoices_file is None:
        invoices_file = data_path / "invoices_processed.csv"
        if not invoices_file.exists():
            invoices_file = data_path.parent / "raw" / "invoices.csv"
        logger.info(f"Loading invoices from: {invoices_file}")
    
    COMPANIES = pd.read_csv(companies_file)
    INVOICES = pd.read_csv(invoices_file)
    
    logger.info("Loading graph...")
    # Handle PyTorch 2.6+ safe_globals for torch_geometric
    try:
        GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt", weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load graph with weights_only=False: {e}")
        try:
            from torch_geometric.data import Data as PyGData
            import torch.serialization
            torch.serialization.add_safe_globals([PyGData])
            GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt", weights_only=False)
        except Exception as e2:
            logger.error(f"Failed to load graph: {e2}")
            # Try loading without weights_only parameter
            try:
                GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt")
            except Exception as e3:
                logger.error(f"Failed to load graph with all methods: {e3}")
                raise
    
    # Load NetworkX graph for easier manipulation
    try:
        with open(data_path / "graphs" / "networkx_graph.gpickle", "rb") as f:
            NETWORKX_GRAPH = pickle.load(f)
        logger.info("Loaded NetworkX graph")
    except Exception as e:
        logger.warning(f"Could not load NetworkX graph: {e}")
        NETWORKX_GRAPH = nx.DiGraph()
    
    logger.info("Loading model...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = GNNFraudDetector(in_channels=3, hidden_channels=64, out_channels=2, model_type="gcn").to(DEVICE)
    
    try:
        MODEL.load_state_dict(torch.load(models_path / "best_model.pt", map_location=DEVICE))
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model weights: {e}")
    
    with open(data_path / "graphs" / "node_mappings.pkl", "rb") as f:
        MAPPINGS = pickle.load(f)
    
    # Get fraud predictions
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(GRAPH_DATA.x.to(DEVICE), GRAPH_DATA.edge_index.to(DEVICE))
        predictions = torch.softmax(out, dim=1)
        FRAUD_PROBA = predictions[:, 1].cpu().numpy()
    
    # Handle mismatch between companies data and graph nodes
    # This can happen when companies data has been updated but graph hasn't been rebuilt
    if len(COMPANIES) != len(FRAUD_PROBA):
        logger.warning(f"Length mismatch: Companies ({len(COMPANIES)}) vs Fraud probabilities ({len(FRAUD_PROBA)})")
        if len(COMPANIES) > len(FRAUD_PROBA):
            # More companies than graph nodes - truncate companies to match
            logger.info(f"Truncating companies data from {len(COMPANIES)} to {len(FRAUD_PROBA)} rows")
            COMPANIES = COMPANIES.iloc[:len(FRAUD_PROBA)]
        else:
            # More graph nodes than companies - pad fraud probabilities with zeros
            logger.info(f"Padding fraud probabilities from {len(FRAUD_PROBA)} to {len(COMPANIES)} elements")
            padded_fraud_proba = np.zeros(len(COMPANIES))
            padded_fraud_proba[:len(FRAUD_PROBA)] = FRAUD_PROBA
            FRAUD_PROBA = padded_fraud_proba
    
    COMPANIES["fraud_probability"] = FRAUD_PROBA
    COMPANIES["predicted_fraud"] = (FRAUD_PROBA > 0.5).astype(int)
    
    logger.info("Model and data loaded successfully!")


@app.route('/upload', methods=['GET'])
def upload_page():
    """Render upload page"""
    return render_template('upload.html')


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Accept CSV uploads (companies or invoices), save and record metadata"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file part'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'no selected file'}), 400

        fname = f.filename
        # Save under data/uploads/<timestamp>/fname
        uploads_dir = Path(__file__).parent / 'data' / 'uploads' / time.strftime('%Y%m%d')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / fname
        f.save(str(save_path))

        # Quick validation: try reading head with pandas
        import pandas as pd
        df = pd.read_csv(save_path)
        rows, cols = df.shape

        # Optional encryption: form field 'encrypt' can be 'on'/'true'/'1'
        encrypt_flag = str(request.form.get('encrypt', '')).lower() in ('1', 'true', 'on', 'yes')
        stored_path = save_path
        encrypted = 0
        if encrypt_flag:
            try:
                enc_path = encrypt_file(save_path)
                # remove plaintext copy
                try:
                    import os
                    os.remove(save_path)
                except Exception:
                    pass
                stored_path = enc_path
                encrypted = 1
            except Exception as ee:
                logger.error(f"Encryption failed: {ee}", exc_info=True)
                return jsonify({'error': 'encryption_failed', 'detail': str(ee)}), 500

        # Record to DB (include encrypted flag)
        record_upload(fname, stored_path, uploader=request.form.get('uploader', 'anonymous'), filetype='csv', rows=int(rows), columns=int(cols), encrypted=encrypted)
        
        # Process the uploaded CSV for incremental learning
        incremental_results = None
        try:
            logger.info(f"Starting incremental learning for file: {fname}")
            incremental_results = process_incremental_learning(save_path, fname)
            logger.info(f"Incremental learning completed for file: {fname}")
        except Exception as e:
            logger.error(f"Incremental learning failed for {fname}: {e}", exc_info=True)
            # Don't fail the upload if incremental learning fails
            pass

        response_data = {
            'status': 'ok', 
            'filename': fname, 
            'rows': int(rows), 
            'columns': int(cols), 
            'encrypted': bool(encrypted)
        }
        
        # Add incremental learning results if available
        if incremental_results:
            response_data['incremental_learning'] = incremental_results
            
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def process_incremental_learning(file_path, filename):
    """
    Process uploaded CSV for incremental learning
    """
    global COMPANIES, INVOICES, NETWORKX_GRAPH
    
    logger.info(f"Starting incremental learning for {filename}")
    
    # Load the uploaded data
    df = pd.read_csv(file_path)
    
    # Determine if it's companies or invoices data
    new_companies_df = pd.DataFrame()
    new_invoices_df = pd.DataFrame()
    
    if "company_id" in df.columns:
        # Companies data
        logger.info("Processing companies data...")
        new_companies_df = df.copy()
        # Apply basic cleaning (similar to clean_data.py)
        if "turnover" not in new_companies_df.columns:
            new_companies_df["turnover"] = 0
        if "location" not in new_companies_df.columns:
            new_companies_df["location"] = "Unknown"
        if "is_fraud" not in new_companies_df.columns:
            new_companies_df["is_fraud"] = 0
            
        # Ensure correct data types
        new_companies_df["company_id"] = new_companies_df["company_id"].astype(str).str.strip()
        new_companies_df["turnover"] = pd.to_numeric(new_companies_df["turnover"], errors='coerce').fillna(0)
        new_companies_df["is_fraud"] = pd.to_numeric(new_companies_df["is_fraud"], errors='coerce').fillna(0).astype(int)
    elif "seller_id" in df.columns and "buyer_id" in df.columns:
        # Invoices data
        logger.info("Processing invoices data...")
        new_invoices_df = df.copy()
        # Apply basic cleaning
        if "amount" not in new_invoices_df.columns:
            new_invoices_df["amount"] = 0
        if "itc_claimed" not in new_invoices_df.columns:
            new_invoices_df["itc_claimed"] = 0
            
        # Ensure correct data types
        new_invoices_df["seller_id"] = new_invoices_df["seller_id"].astype(str).str.strip()
        new_invoices_df["buyer_id"] = new_invoices_df["buyer_id"].astype(str).str.strip()
        new_invoices_df["amount"] = pd.to_numeric(new_invoices_df["amount"], errors='coerce').fillna(0)
        new_invoices_df["itc_claimed"] = pd.to_numeric(new_invoices_df["itc_claimed"], errors='coerce').fillna(0)
    else:
        logger.warning(f"Unknown data format in {filename}")
        return
    
    # If we have invoices but no companies data, we need to extract company info
    if not new_companies_df.empty and new_invoices_df.empty:
        # Companies data only - need to engineer features
        logger.info("Engineering features for new companies...")
        # Simple feature engineering for new companies
        new_companies_df["sent_invoice_count"] = 0
        new_companies_df["received_invoice_count"] = 0
        new_companies_df["total_sent_amount"] = 0
        new_companies_df["total_received_amount"] = 0
    elif new_companies_df.empty and not new_invoices_df.empty:
        # Invoices data only - extract company info
        logger.info("Extracting company info from invoices...")
        seller_info = new_invoices_df.groupby("seller_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        seller_info.columns = ["company_id", "total_sent_amount", "sent_invoice_count"]
        seller_info["company_id"] = seller_info["company_id"].astype(str)
        
        buyer_info = new_invoices_df.groupby("buyer_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        buyer_info.columns = ["company_id", "total_received_amount", "received_invoice_count"]
        buyer_info["company_id"] = buyer_info["company_id"].astype(str)
        
        # Merge seller and buyer info
        company_info = pd.merge(seller_info, buyer_info, on="company_id", how="outer").fillna(0)
        company_info["turnover"] = company_info["total_sent_amount"] + company_info["total_received_amount"]
        company_info["is_fraud"] = 0  # Default to non-fraud
        
        new_companies_df = company_info
    elif not new_companies_df.empty and not new_invoices_df.empty:
        # Both companies and invoices - engineer features
        logger.info("Engineering features for companies with invoices...")
        # Count invoices sent (seller perspective)
        seller_counts = new_invoices_df.groupby("seller_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        seller_counts.columns = ["company_id", "total_sent_amount", "sent_invoice_count"]
        seller_counts["company_id"] = seller_counts["company_id"].astype(str)
        
        # Count invoices received (buyer perspective)
        buyer_counts = new_invoices_df.groupby("buyer_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        buyer_counts.columns = ["company_id", "total_received_amount", "received_invoice_count"]
        buyer_counts["company_id"] = buyer_counts["company_id"].astype(str)
        
        # Merge features back to companies
        new_companies_df = new_companies_df.merge(seller_counts, on="company_id", how="left")
        new_companies_df = new_companies_df.merge(buyer_counts, on="company_id", how="left")
        
        # Fill NaN with 0
        new_companies_df.fillna(0, inplace=True)
    
    # Update graph with new data (without converting company_id to int)
    NETWORKX_GRAPH, MAPPINGS, nodes_added, edges_added = update_graph(new_companies_df, new_invoices_df)
    
    # Identify affected nodes
    affected_nodes = identify_affected_nodes(new_companies_df, new_invoices_df, k_hop=2)
    
    # Extract subgraph
    subgraph = extract_subgraph(affected_nodes, k_hop=2)
    
    # Get subgraph statistics
    subgraph_nodes = subgraph.number_of_nodes()
    subgraph_edges = subgraph.number_of_edges()
    
    # Identify central nodes (nodes with highest degree)
    central_nodes = []
    if subgraph_nodes > 0:
        node_degrees = [(node, subgraph.degree(node)) for node in subgraph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        central_nodes = [node for node, degree in node_degrees[:min(5, len(node_degrees))]]
    
    # Count high-risk nodes in subgraph
    high_risk_count = 0
    fraud_node_count = 0
    if hasattr(subgraph, 'nodes') and len(subgraph.nodes()) > 0:
        for node in subgraph.nodes():
            if node in NETWORKX_GRAPH.nodes():
                node_data = NETWORKX_GRAPH.nodes[node]
                if 'is_fraud' in node_data and node_data['is_fraud'] == 1:
                    fraud_node_count += 1
                # This would be updated after retraining, but we can estimate based on existing data
    
    # Convert subgraph to PyTorch Geometric format
    subgraph_data, _, _ = networkx_to_pytorch_geometric_subgraph(subgraph, MAPPINGS["node_to_idx"])
    
    # Perform incremental retraining
    updated_model_state = incremental_retrain(subgraph_data, epochs=50, lr=0.001)
    
    # Update global embeddings
    update_global_embeddings()
    
    # Save updated graph and model
    save_updated_graph_and_model()
    
    logger.info(f"Incremental learning completed successfully - {nodes_added} nodes and {edges_added} edges added")
    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "affected_nodes": len(affected_nodes),
        "subgraph_nodes": subgraph_nodes,
        "subgraph_edges": subgraph_edges,
        "central_nodes_count": len(central_nodes),
        "high_risk_nodes": high_risk_count,
        "fraud_nodes": fraud_node_count
    }


def update_graph(new_companies_df, new_invoices_df):
    """
    Update the existing graph with new nodes and edges from uploaded data
    Returns updated NetworkX graph and mappings
    """
    global NETWORKX_GRAPH, MAPPINGS, COMPANIES, INVOICES
    
    logger.info("Updating graph with new data...")
    
    # Count initial nodes and edges
    initial_nodes = NETWORKX_GRAPH.number_of_nodes() if NETWORKX_GRAPH else 0
    initial_edges = NETWORKX_GRAPH.number_of_edges() if NETWORKX_GRAPH else 0
    
    # Update companies data
    COMPANIES = pd.concat([COMPANIES, new_companies_df], ignore_index=True)
    COMPANIES = COMPANIES.drop_duplicates(subset=["company_id"], keep="last")
    
    # Update invoices data
    INVOICES = pd.concat([INVOICES, new_invoices_df], ignore_index=True)
    
    # Add new nodes to NetworkX graph
    nodes_added = 0
    for _, row in new_companies_df.iterrows():
        try:
            company_id = row["company_id"]  # Keep as string
            if company_id not in NETWORKX_GRAPH.nodes():
                NETWORKX_GRAPH.add_node(
                    company_id,  # Keep as string
                    turnover=float(row.get("turnover", 0)),
                    location=str(row.get("location", "Unknown")),
                    is_fraud=int(row.get("is_fraud", 0)),
                    sent_invoices=float(row.get("sent_invoice_count", 0)),
                    received_invoices=float(row.get("received_invoice_count", 0)),
                    total_sent_amount=float(row.get("total_sent_amount", 0)),
                    total_received_amount=float(row.get("total_received_amount", 0))
                )
                nodes_added += 1
        except Exception as e:
            logger.warning(f"Error adding node {row.get('company_id')}: {e}")
            continue
    
    # Add new edges to NetworkX graph
    edges_added = 0
    for _, row in new_invoices_df.iterrows():
        try:
            seller = row["seller_id"]  # Keep as string
            buyer = row["buyer_id"]    # Keep as string
            
            # Only add edge if both nodes exist
            if seller in NETWORKX_GRAPH and buyer in NETWORKX_GRAPH:
                # Check if edge already exists to avoid duplicates
                if not NETWORKX_GRAPH.has_edge(seller, buyer):
                    NETWORKX_GRAPH.add_edge(
                        seller,
                        buyer,
                        amount=float(row.get("amount", 0)),
                        itc_claimed=float(row.get("itc_claimed", 0))
                    )
                    edges_added += 1
        except Exception as e:
            logger.warning(f"Error adding edge: {e}")
            continue
    
    # Update mappings
    node_list = sorted(list(NETWORKX_GRAPH.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    MAPPINGS = {
        "node_list": node_list,
        "node_to_idx": node_to_idx
    }
    
    final_nodes = NETWORKX_GRAPH.number_of_nodes()
    final_edges = NETWORKX_GRAPH.number_of_edges()
    
    logger.info(f"Graph updated: {final_nodes} nodes (+{nodes_added}), {final_edges} edges (+{edges_added})")
    return NETWORKX_GRAPH, MAPPINGS, nodes_added, edges_added


def identify_affected_nodes(new_companies_df, new_invoices_df, k_hop=2):
    """
    Identify nodes that are affected by new data (new nodes + neighbors within k-hop)
    Returns list of affected node IDs
    """
    global NETWORKX_GRAPH
    
    logger.info("Identifying affected nodes...")
    
    # Start with new nodes
    new_node_ids = set()
    for _, row in new_companies_df.iterrows():
        try:
            new_node_ids.add(row["company_id"])  # Keep as string
        except:
            continue
    
    # Add nodes connected through new edges
    for _, row in new_invoices_df.iterrows():
        try:
            seller = row["seller_id"]  # Keep as string
            buyer = row["buyer_id"]    # Keep as string
            new_node_ids.add(seller)
            new_node_ids.add(buyer)
        except:
            continue
    
    # Find k-hop neighbors using BFS
    affected_nodes = set(new_node_ids)
    queue = deque([(node, 0) for node in new_node_ids])  # (node, distance)
    visited = set(new_node_ids)
    
    while queue:
        node, distance = queue.popleft()
        
        if distance < k_hop:
            # Get neighbors (both incoming and outgoing)
            neighbors = set(NETWORKX_GRAPH.successors(node)) | set(NETWORKX_GRAPH.predecessors(node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    affected_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    logger.info(f"Identified {len(affected_nodes)} affected nodes within {k_hop}-hop neighborhood")
    return list(affected_nodes)


def extract_subgraph(affected_nodes, k_hop=2):
    """
    Extract k-hop subgraph around affected nodes
    Returns NetworkX subgraph
    """
    global NETWORKX_GRAPH
    
    logger.info(f"Extracting {k_hop}-hop subgraph for {len(affected_nodes)} nodes...")
    
    # Get k-hop neighborhood using BFS
    subgraph_nodes = set(affected_nodes)
    queue = deque([(node, 0) for node in affected_nodes])  # (node, distance)
    visited = set(affected_nodes)
    
    while queue:
        node, distance = queue.popleft()
        
        if distance < k_hop:
            # Get neighbors (both incoming and outgoing)
            neighbors = set(NETWORKX_GRAPH.successors(node)) | set(NETWORKX_GRAPH.predecessors(node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    # Extract subgraph
    subgraph = NETWORKX_GRAPH.subgraph(subgraph_nodes).copy()
    logger.info(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    
    return subgraph


def networkx_to_pytorch_geometric_subgraph(G, full_node_to_idx):
    """
    Convert NetworkX subgraph to PyTorch Geometric Data object
    """
    logger.info("Converting subgraph to PyTorch Geometric format...")
    
    # Create node list and feature matrix for subgraph
    node_list = sorted(list(G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    x_list = []
    y_list = []
    
    for node in node_list:
        node_data = G.nodes[node]
        features = [
            node_data.get("turnover", 0),
            node_data.get("sent_invoices", 0),
            node_data.get("received_invoices", 0)
        ]
        x_list.append(features)
        y_list.append(node_data.get("is_fraud", 0))
    
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    
    logger.info(f"Subgraph node feature matrix shape: {x.shape}")
    
    # Create edge indices and attributes
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_index.append([u_idx, v_idx])
        edge_attr.append([data.get("amount", 0)])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    logger.info(f"Subgraph edge index shape: {edge_index.shape}")
    
    # Create PyG Data object (without node_ids since they're strings)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    
    return data, node_list, node_to_idx


def incremental_retrain(subgraph_data, epochs=50, lr=0.001):
    """
    Retrain model on subgraph data
    Returns updated model state dict
    """
    global MODEL, DEVICE
    
    logger.info("Starting incremental retraining on subgraph...")
    
    # Move data to device
    subgraph_data = subgraph_data.to(DEVICE)
    
    # Create optimizer (use same parameters as original training)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Set model to training mode
    MODEL.train()
    
    # Train for specified epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = MODEL(subgraph_data.x, subgraph_data.edge_index)
        loss = criterion(out, subgraph_data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Incremental retraining epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")
    
    # Set model back to eval mode
    MODEL.eval()
    
    logger.info("Incremental retraining completed")
    return MODEL.state_dict()


def update_global_embeddings():
    """
    Update global fraud probabilities for all nodes after incremental training
    """
    global MODEL, GRAPH_DATA, DEVICE, FRAUD_PROBA, COMPANIES
    
    logger.info("Updating global embeddings and fraud probabilities...")
    
    # Get updated predictions for all nodes
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(GRAPH_DATA.x.to(DEVICE), GRAPH_DATA.edge_index.to(DEVICE))
        predictions = torch.softmax(out, dim=1)
        FRAUD_PROBA = predictions[:, 1].cpu().numpy()
    
    # Update COMPANIES dataframe
    COMPANIES["fraud_probability"] = FRAUD_PROBA
    COMPANIES["predicted_fraud"] = (FRAUD_PROBA > 0.5).astype(int)
    
    logger.info("Global embeddings updated")


def save_updated_graph_and_model():
    """
    Save updated graph, mappings, and model weights
    """
    global GRAPH_DATA, MAPPINGS, MODEL, NETWORKX_GRAPH
    
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    graph_path = data_path / "graphs"
    
    # Ensure the graphs directory exists
    graph_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving updated graph and model...")
    
    # Save NetworkX graph using pickle (NetworkX 3.x compatibility)
    with open(graph_path / "networkx_graph.gpickle", "wb") as f:
        pickle.dump(NETWORKX_GRAPH, f)
    logger.info("✓ NetworkX graph saved")
    
    # Convert NetworkX graph to PyTorch Geometric and save
    try:
        from src.graph_construction.build_graph import GraphBuilder
        builder = GraphBuilder(str(data_path))  # Pass the correct path
        pyg_data, node_list, node_to_idx = builder.networkx_to_pytorch_geometric(NETWORKX_GRAPH, COMPANIES)
        torch.save(pyg_data, graph_path / "graph_data.pt")
        logger.info("✓ PyTorch Geometric graph saved")
        
        # Update global GRAPH_DATA
        GRAPH_DATA = pyg_data
        
        # Save mappings
        mappings = {
            "node_list": node_list,
            "node_to_idx": node_to_idx
        }
        with open(graph_path / "node_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        logger.info("✓ Node mappings saved")
        
        MAPPINGS = mappings
    except Exception as e:
        logger.error(f"Error converting/saving PyTorch Geometric graph: {e}")
    
    # Save updated model
    try:
        torch.save(MODEL.state_dict(), models_path / "best_model.pt")
        logger.info("✓ Model weights saved")
    except Exception as e:
        logger.error(f"Error saving model weights: {e}")


@app.route('/uploads')
def uploads_list():
    try:
        init_db()
        items = list_uploads(limit=100)
        return jsonify(items)
    except Exception as e:
        logger.error(f"Error listing uploads: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTES - Pages
# ============================================================================

@app.route("/")
def home():
    """Serve landing page as default"""
    return render_template('landing.html')


# Serve static files (CSS, JS, images) from static folder
@app.route("/static/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# Serve React assets if React build exists
@app.route("/assets/<path:path>")
def serve_react_assets(path):
    """Serve React static assets (JS, CSS, etc.)"""
    react_static_path = Path(__file__).parent / "static" / "react" / "assets"
    if (react_static_path / path).exists():
        return send_from_directory(str(react_static_path), path)
    return "Not Found", 404


@app.route("/dashboard")
def dashboard():
    """Dashboard route - serves React or template"""
    react_build_path = Path(__file__).parent / "static" / "react" / "index.html"
    if react_build_path.exists():
        return send_from_directory(str(react_build_path.parent), "index.html")
    # Fallback to template
    high_risk_count = (FRAUD_PROBA > 0.5).sum()
    avg_risk = FRAUD_PROBA.mean()
    fraud_count = (COMPANIES["predicted_fraud"] == 1).sum()
    
    return render_template("index.html",
                         total_companies=len(COMPANIES),
                         high_risk_count=int(high_risk_count),
                         fraud_count=int(fraud_count),
                         avg_risk=f"{avg_risk:.2%}")


@app.route("/companies")
def companies():
    """Companies route - serves React or template"""
    react_build_path = Path(__file__).parent / "static" / "react" / "index.html"
    if react_build_path.exists():
        return send_from_directory(str(react_build_path.parent), "index.html")
    return render_template("companies.html")


@app.route("/analytics")
def analytics():
    """Analytics route - serves React or template"""
    react_build_path = Path(__file__).parent / "static" / "react" / "index.html"
    if react_build_path.exists():
        return send_from_directory(str(react_build_path.parent), "index.html")
    return render_template("analytics.html")


@app.route('/chatbot')
def chatbot_page():
    """Render chatbot page"""
    return render_template('chatbot.html')

@app.route('/landing')
def landing_page():
    """Render modern landing page"""
    return render_template('landing.html')

# ============================================================================
# ROUTES - API
# ============================================================================

@app.route("/api/companies")
def get_companies():
    """API: Get all companies with filters"""
    try:
        # Get filters from query parameters
        risk_threshold = float(request.args.get("risk_threshold", 0.5))
        location_filter = request.args.get("location", "").split(",")
        location_filter = [l.strip() for l in location_filter if l.strip()]
        search_id = request.args.get("search", "").strip()
        
        # Filter data
        filtered_df = COMPANIES.copy()
        
        # Apply location filter
        if location_filter and location_filter != [""]:
            filtered_df = filtered_df[filtered_df["location"].isin(location_filter)]
        
        # Apply risk threshold
        filtered_df = filtered_df[filtered_df["fraud_probability"] >= risk_threshold]
        
        # Apply search (company_id is a string like "GST000123")
        if search_id:
            filtered_df = filtered_df[filtered_df["company_id"].astype(str).str.contains(search_id, case=False)]
        
        # Sort by fraud probability
        filtered_df = filtered_df.sort_values("fraud_probability", ascending=False)
        
        # Format for JSON
        companies_list = []
        for _, row in filtered_df.iterrows():
            companies_list.append({
                "company_id": str(row["company_id"]),
                "location": str(row["location"]),
                "turnover": f"₹{float(row['turnover']):.2f}",
                "fraud_probability": f"{float(row['fraud_probability']):.2%}",
                "risk_level": "🔴 HIGH" if float(row["fraud_probability"]) > 0.7 else "🟡 MEDIUM" if float(row["fraud_probability"]) > 0.3 else "🟢 LOW",
                "status": "🚨 FRAUD" if int(row["predicted_fraud"]) == 1 else "✅ NORMAL"
            })
        
        return jsonify(companies_list)
    
    except Exception as e:
        logger.error(f"Error in get_companies: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/company/<company_id>")
def get_company_detail(company_id):
    """API: Get company details"""
    try:
        company_id_str = str(company_id).strip()
        company = COMPANIES[COMPANIES["company_id"].astype(str) == company_id_str]
        
        if len(company) == 0:
            return jsonify({"error": f"Company {company_id_str} not found"}), 404
        
        row = company.iloc[0]
        
        # Get transaction info
        outgoing = INVOICES[INVOICES["seller_id"].astype(str) == company_id_str]
        incoming = INVOICES[INVOICES["buyer_id"].astype(str) == company_id_str]
        
        return jsonify({
            "company_id": company_id_str,
            "location": str(row["location"]),
            "turnover": float(row["turnover"]),
            "fraud_probability": float(row["fraud_probability"]),
            "predicted_fraud": int(row["predicted_fraud"]),
            "risk_level": "HIGH" if float(row["fraud_probability"]) > 0.7 else "MEDIUM" if float(row["fraud_probability"]) > 0.3 else "LOW",
            "sent_invoice_count": int(row.get("sent_invoices", 0)),
            "received_invoice_count": int(row.get("received_invoices", 0)),
            "total_sent_amount": float(row.get("total_sent_amount", 0)),
            "total_received_amount": float(row.get("total_received_amount", 0)),
            "outgoing_invoices": len(outgoing),
            "incoming_invoices": len(incoming)
        })
    
    except Exception as e:
        logger.error(f"Error in get_company_detail: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/statistics")
def get_statistics():
    """API: Get overall statistics"""
    try:
        # Check if data is loaded
        if FRAUD_PROBA is None or len(FRAUD_PROBA) == 0:
            logger.error("FRAUD_PROBA is not initialized")
            return jsonify({"error": "Model not loaded", "total_companies": 0}), 500
        
        if GRAPH_DATA is None:
            logger.error("GRAPH_DATA is not initialized")
            return jsonify({"error": "Graph data not loaded", "total_companies": 0}), 500
        
        if COMPANIES is None or len(COMPANIES) == 0:
            logger.error("COMPANIES is not initialized")
            return jsonify({"error": "Companies data not loaded", "total_companies": 0}), 500
        
        high_risk = (FRAUD_PROBA > 0.7).sum()
        medium_risk = ((FRAUD_PROBA > 0.3) & (FRAUD_PROBA <= 0.7)).sum()
        low_risk = (FRAUD_PROBA <= 0.3).sum()
        
        stats = {
            "total_companies": len(COMPANIES),
            "total_edges": int(GRAPH_DATA.num_edges),
            "high_risk_count": int(high_risk),
            "medium_risk_count": int(medium_risk),
            "low_risk_count": int(low_risk),
            "fraud_count": int((COMPANIES["predicted_fraud"] == 1).sum()),
            "average_fraud_probability": float(np.mean(FRAUD_PROBA))
        }
        
        logger.info(f"Returning statistics: total_companies={stats['total_companies']}, total_edges={stats['total_edges']}")
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}", exc_info=True)
        return jsonify({"error": str(e), "total_companies": 0}), 500


@app.route("/api/chart/fraud_distribution")
def chart_fraud_distribution():
    """API: Fraud distribution chart - returns Plotly JSON"""
    try:
        fraud_dist = COMPANIES["predicted_fraud"].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=["Normal", "Fraud"],
                   y=[fraud_dist.get(0, 0), fraud_dist.get(1, 0)],
                   marker=dict(color=["green", "red"]))
        ])
        fig.update_layout(
            title="Fraud Distribution",
            xaxis_title="Status",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_fraud_distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/risk_distribution")
def chart_risk_distribution():
    """API: Risk score distribution chart - returns Plotly JSON"""
    try:
        fig = go.Figure(data=[
            go.Histogram(
                x=FRAUD_PROBA.tolist(),  # Convert numpy array to list
                nbinsx=30,
                marker=dict(color="blue")
            )
        ])
        fig.update_layout(
            title="Fraud Probability Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            height=400
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                     annotation_text="Threshold: 0.50")
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_risk_distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/risk_by_location")
def chart_risk_by_location():
    """API: Risk by location chart - returns Plotly JSON"""
    try:
        fig = px.box(
            COMPANIES,
            x="location",
            y="fraud_probability",
            title="Fraud Probability Distribution by Location",
            labels={"fraud_probability": "Fraud Probability", "location": "Location"}
        )
        fig.update_layout(height=400)
        
        # Serialize with custom encoder
        fig_dict = fig.to_dict()
        return Response(json.dumps(fig_dict, cls=NumpyEncoder), mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error in chart_risk_by_location: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/turnover_vs_risk")
def chart_turnover_vs_risk():
    """API: Turnover vs Risk scatter plot - returns Plotly JSON"""
    try:
        fig = px.scatter(
            COMPANIES,
            x="turnover",
            y="fraud_probability",
            color="predicted_fraud",
            title="Company Turnover vs Fraud Risk",
            labels={"turnover": "Turnover (₹)", "fraud_probability": "Fraud Probability"},
            hover_data=["company_id", "location"]
        )
        fig.update_layout(height=400)
        
        # Serialize with custom encoder
        fig_dict = fig.to_dict()
        return Response(json.dumps(fig_dict, cls=NumpyEncoder), mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error in chart_turnover_vs_risk: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/top_senders_table")
def get_top_senders_table():
    """API: Top invoice senders as table data (JSON)"""
    try:
        # Check if seller_id column exists, if not use first column as sender
        sender_col = "seller_id" if "seller_id" in INVOICES.columns else INVOICES.columns[0]
        logger.info(f"Using column '{sender_col}' for sellers")
        
        top_senders = INVOICES.groupby(sender_col).size().nlargest(10)
        logger.info(f"Found {len(top_senders)} top senders")
        
        data = []
        for sender_id, count in top_senders.items():
            sender_id_str = str(sender_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == sender_id_str]
            if len(company) > 0:
                fraud_prob = float(company.iloc[0].get('fraud_probability', 0))
                data.append({
                    "company_id": sender_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{fraud_prob:.2%}"
                })
        
        logger.info(f"Returning {len(data)} senders with matching companies")
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_senders_table: {e}", exc_info=True)
        return jsonify({"error": str(e), "invoices_cols": list(INVOICES.columns)}), 500


@app.route("/api/top_receivers_table")
def get_top_receivers_table():
    """API: Top invoice receivers as table data (JSON)"""
    try:
        # Check if buyer_id column exists, if not use second column as receiver
        receiver_col = "buyer_id" if "buyer_id" in INVOICES.columns else (INVOICES.columns[1] if len(INVOICES.columns) > 1 else INVOICES.columns[0])
        logger.info(f"Using column '{receiver_col}' for receivers")
        
        top_receivers = INVOICES.groupby(receiver_col).size().nlargest(10)
        logger.info(f"Found {len(top_receivers)} top receivers")
        
        data = []
        for receiver_id, count in top_receivers.items():
            receiver_id_str = str(receiver_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == receiver_id_str]
            if len(company) > 0:
                fraud_prob = float(company.iloc[0].get('fraud_probability', 0))
                data.append({
                    "company_id": receiver_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{fraud_prob:.2%}"
                })
        
        logger.info(f"Returning {len(data)} receivers with matching companies")
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_receivers_table: {e}", exc_info=True)
        return jsonify({"error": str(e), "invoices_cols": list(INVOICES.columns)}), 500


@app.route("/api/top_senders")
def get_top_senders():
    """API: Top invoice senders - returns Plotly chart data"""
    try:
        if INVOICES is None or len(INVOICES) == 0:
            logger.warning("INVOICES is empty")
            return jsonify({"error": "No invoice data available"}), 404
        
        if "seller_id" not in INVOICES.columns:
            logger.error(f"Missing seller_id column. Available columns: {list(INVOICES.columns)}")
            return jsonify({"error": f"Missing seller_id column. Available: {list(INVOICES.columns)}"}), 400
        
        top_senders = INVOICES.groupby("seller_id").size().nlargest(10)
        
        if len(top_senders) == 0:
            logger.warning("No top senders found after grouping")
            return jsonify({"error": "No sender data available"}), 404
        
        company_ids = []
        counts = []
        colors = []
        
        for seller_id, count in top_senders.items():
            seller_id_str = str(seller_id).strip()
            company_ids.append(seller_id_str)
            counts.append(int(count))
            
            # Try to find matching company for color coding
            if COMPANIES is not None and len(COMPANIES) > 0 and "company_id" in COMPANIES.columns:
                company = COMPANIES[COMPANIES["company_id"].astype(str).str.strip() == seller_id_str]
                if len(company) > 0 and 'fraud_probability' in company.columns:
                    fraud_prob = float(company.iloc[0]['fraud_probability'])
                    if fraud_prob > 0.7:
                        colors.append('#FF4444')  # Red for high risk
                    elif fraud_prob > 0.3:
                        colors.append('#FF9932')  # Orange for medium risk
                    else:
                        colors.append('#114C5A')  # Blue for low risk
                else:
                    colors.append('#6C757D')  # Gray for unknown
            else:
                colors.append('#114C5A')  # Default color
        
        # Ensure we have data
        if len(company_ids) == 0:
            logger.error("No company IDs collected")
            return jsonify({"error": "No data to display"}), 404
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_ids,
                y=counts,
                marker=dict(color=colors if len(colors) == len(counts) else '#114C5A'),
                text=counts,
                textposition='outside',
                textfont=dict(size=11)
            )
        ])
        fig.update_layout(
            title="Top 10 Invoice Senders",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#172B36', family='Inter, sans-serif'),
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max(counts) * 1.15] if counts else [0, 10]),
            margin=dict(b=100, l=60, r=20, t=60)
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in get_top_senders: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": str(e.__traceback__)}), 500


@app.route("/api/top_receivers")
def get_top_receivers():
    """API: Top invoice receivers - returns Plotly chart data"""
    try:
        if INVOICES is None or len(INVOICES) == 0:
            logger.warning("INVOICES is empty")
            return jsonify({"error": "No invoice data available"}), 404
        
        if "buyer_id" not in INVOICES.columns:
            logger.error(f"Missing buyer_id column. Available columns: {list(INVOICES.columns)}")
            return jsonify({"error": f"Missing buyer_id column. Available: {list(INVOICES.columns)}"}), 400
        
        top_receivers = INVOICES.groupby("buyer_id").size().nlargest(10)
        
        if len(top_receivers) == 0:
            logger.warning("No top receivers found after grouping")
            return jsonify({"error": "No receiver data available"}), 404
        
        company_ids = []
        counts = []
        colors = []
        
        for buyer_id, count in top_receivers.items():
            buyer_id_str = str(buyer_id).strip()
            company_ids.append(buyer_id_str)
            counts.append(int(count))
            
            # Try to find matching company for color coding
            if COMPANIES is not None and len(COMPANIES) > 0 and "company_id" in COMPANIES.columns:
                company = COMPANIES[COMPANIES["company_id"].astype(str).str.strip() == buyer_id_str]
                if len(company) > 0 and 'fraud_probability' in company.columns:
                    fraud_prob = float(company.iloc[0]['fraud_probability'])
                    if fraud_prob > 0.7:
                        colors.append('#FF4444')  # Red for high risk
                    elif fraud_prob > 0.3:
                        colors.append('#FF9932')  # Orange for medium risk
                    else:
                        colors.append('#114C5A')  # Blue for low risk
                else:
                    colors.append('#6C757D')  # Gray for unknown
            else:
                colors.append('#FF6B6B')  # Default coral color
        
        # Ensure we have data
        if len(company_ids) == 0:
            logger.error("No company IDs collected")
            return jsonify({"error": "No data to display"}), 404
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_ids,
                y=counts,
                marker=dict(color=colors if len(colors) == len(counts) else '#FF6B6B'),
                text=counts,
                textposition='outside',
                textfont=dict(size=11)
            )
        ])
        fig.update_layout(
            title="Top 10 Invoice Receivers",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#172B36', family='Inter, sans-serif'),
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max(counts) * 1.15] if counts else [0, 10]),
            margin=dict(b=100, l=60, r=20, t=60)
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in get_top_receivers: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": str(e.__traceback__)}), 500


@app.route("/api/locations")
def get_locations():
    """API: Get all unique locations for filtering"""
    try:
        locations = sorted(COMPANIES["location"].unique().tolist())
        return jsonify(locations)
    except Exception as e:
        logger.error(f"Error in get_locations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """API: Single company prediction"""
    try:
        data = request.get_json()
        company_id = data.get("company_id")
        
        if not company_id:
            return jsonify({"error": "company_id is required"}), 400
        
        node_list = MAPPINGS["node_list"]
        if company_id not in node_list:
            return jsonify({"error": f"Company ID {company_id} not found"}), 404
        
        node_idx = node_list.index(company_id)
        fraud_proba = float(FRAUD_PROBA[node_idx])
        
        company_row = COMPANIES[COMPANIES["company_id"] == company_id].iloc[0]
        
        return jsonify({
            "company_id": company_id,
            "fraud_probability": fraud_proba,
            "is_fraud": float(fraud_proba > 0.5),
            "risk_level": "HIGH" if fraud_proba > 0.7 else "MEDIUM" if fraud_proba > 0.3 else "LOW",
            "location": company_row["location"],
            "turnover": float(company_row["turnover"])
        })
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chatbot", methods=['POST'])
def chatbot_api():
    """API endpoint for chatbot queries"""
    try:
        # Get the user message
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Import Groq client
        from groq import Groq
        
        # Initialize Groq client with the provided API key
        GROQ_API_KEY = "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"
        client = Groq(api_key=GROQ_API_KEY)
        
        # Get data statistics for context
        def get_data_statistics():
            stats = []
            
            if COMPANIES is not None:
                stats.append(f"Companies Dataset: {len(COMPANIES)} records")
                if "is_fraud" in COMPANIES.columns:
                    fraud_count = COMPANIES["is_fraud"].sum()
                    stats.append(f"Fraud Companies: {fraud_count} ({fraud_count/len(COMPANIES)*100:.2f}%)")
                if "turnover" in COMPANIES.columns:
                    stats.append(f"Total Turnover: ₹{COMPANIES['turnover'].sum():,.0f}")
                    stats.append(f"Average Turnover: ₹{COMPANIES['turnover'].mean():,.0f}")
                if "location" in COMPANIES.columns:
                    stats.append(f"Locations Covered: {COMPANIES['location'].nunique()}")
                    top_locations = COMPANIES["location"].value_counts().head(3)
                    stats.append(f"Top 3 Locations: {dict(top_locations)}")
            
            if INVOICES is not None:
                stats.append(f"Invoices Dataset: {len(INVOICES)} records")
                if "amount" in INVOICES.columns:
                    stats.append(f"Total Invoice Value: ₹{INVOICES['amount'].sum():,.0f}")
                    stats.append(f"Average Invoice Value: ₹{INVOICES['amount'].mean():,.0f}")
                if "itc_claimed" in INVOICES.columns:
                    stats.append(f"Total ITC Claims: ₹{INVOICES['itc_claimed'].sum():,.0f}")
                    stats.append(f"Average ITC per Invoice: ₹{INVOICES['itc_claimed'].mean():,.0f}")
            
            return "\n".join(stats)
        
        # Enhanced context for the LLM
        system_context = (
            "You are a GST tax compliance and fraud detection expert assistant. "
            "You have access to a dataset of companies and their invoices. "
            "Provide accurate, data-driven responses based on the following information:\n\n"
            f"=== DATASET STATISTICS ===\n{get_data_statistics()}\n\n"
            "Answer the user's question accurately and concisely."
        )
        
        # Prepare messages with context
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message}
        ]
        
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({'response': ai_response})
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Loading model and data...")
    load_model_and_data()
    
    logger.info("Starting Flask application...")
    app.run(debug=True, host="0.0.0.0", port=5000)

