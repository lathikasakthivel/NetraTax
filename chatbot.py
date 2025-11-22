"""
Chatbot for Tax Fraud GNN Application
Integrates with the existing tax-fraud-gnn system to provide conversational insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from groq import Groq

# Configure Streamlit
st.set_page_config(
    page_title="GST Fraud Detection Assistant", 
    page_icon="🕵️", 
    layout="wide"
)

# Add custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .fraud-highlight {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-highlight {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🕵️ GST Fraud Detection Assistant</h1></div>', unsafe_allow_html=True)
st.info("Ask questions about GST companies, invoices, fraud patterns, and tax compliance")

# Initialize Groq client with the provided API key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Load data from the tax-fraud-gnn system
@st.cache_data
def load_tax_data():
    """Load processed data from the tax-fraud-gnn system"""
    try:
        data_path = Path(__file__).parent / "data" / "processed"
        
        # Load companies data
        companies_file = data_path / "companies_processed.csv"
        if companies_file.exists():
            companies_df = pd.read_csv(companies_file)
        else:
            # Fallback to raw data
            companies_file = data_path.parent / "raw" / "companies.csv"
            companies_df = pd.read_csv(companies_file)
        
        # Load invoices data
        invoices_file = data_path / "invoices_processed.csv"
        if invoices_file.exists():
            invoices_df = pd.read_csv(invoices_file)
        else:
            # Fallback to raw data
            invoices_file = data_path.parent / "raw" / "invoices.csv"
            invoices_df = pd.read_csv(invoices_file)
            
        return companies_df, invoices_df
    except Exception as e:
        st.error(f"Error loading tax data: {e}")
        return None, None

# Load model information (for context)
@st.cache_data
def load_model_info():
    """Load model information for context"""
    try:
        data_path = Path(__file__).parent / "data" / "processed"
        graphs_path = data_path / "graphs"
        
        # Check if graph data exists
        graph_exists = (graphs_path / "graph_data.pt").exists()
        networkx_exists = (graphs_path / "networkx_graph.gpickle").exists()
        
        return {
            "graph_exists": graph_exists,
            "networkx_exists": networkx_exists,
            "model_trained": (Path(__file__).parent / "models" / "best_model.pt").exists()
        }
    except:
        return {
            "graph_exists": False,
            "networkx_exists": False,
            "model_trained": False
        }

# Load data
companies_df, invoices_df = load_tax_data()
model_info = load_model_info()

# Sidebar with dataset information
with st.sidebar:
    st.header("📊 Dataset Overview")
    
    if companies_df is not None:
        st.subheader("Companies")
        st.write(f"**Total Companies:** {len(companies_df):,}")
        
        if "is_fraud" in companies_df.columns:
            fraud_count = companies_df["is_fraud"].sum()
            st.write(f"**Fraud Cases:** {fraud_count:,}")
            st.write(f"**Fraud Rate:** {fraud_count/len(companies_df)*100:.2f}%")
            
        if "turnover" in companies_df.columns:
            total_turnover = companies_df["turnover"].sum()
            st.write(f"**Total Turnover:** ₹{total_turnover:,.0f}")
            
        if "location" in companies_df.columns:
            locations = companies_df["location"].nunique()
            st.write(f"**Locations:** {locations}")
    
    if invoices_df is not None:
        st.subheader("Invoices")
        st.write(f"**Total Invoices:** {len(invoices_df):,}")
        
        if "amount" in invoices_df.columns:
            total_value = invoices_df["amount"].sum()
            st.write(f"**Total Value:** ₹{total_value:,.0f}")
            
        if "itc_claimed" in invoices_df.columns:
            total_itc = invoices_df["itc_claimed"].sum()
            st.write(f"**Total ITC Claims:** ₹{total_itc:,.0f}")
    
    st.divider()
    
    st.subheader("🤖 Model Status")
    st.write(f"**Graph Built:** {'✅' if model_info['graph_exists'] else '❌'}")
    st.write(f"**NetworkX Graph:** {'✅' if model_info['networkx_exists'] else '❌'}")
    st.write(f"**Model Trained:** {'✅' if model_info['model_trained'] else '❌'}")
    
    st.divider()
    
    st.subheader("💡 Example Questions")
    st.markdown("""
    - Show me companies with highest fraud probability
    - Which companies have the largest ITC claims?
    - Find invoices above ₹1,00,000
    - Show fraud distribution by location
    - What's the average turnover of fraudulent companies?
    - List top 10 companies by invoice volume
    - Show correlation between turnover and fraud
    """)

# Function to get data statistics for context
def get_data_statistics():
    """Get comprehensive statistics about the dataset for LLM context"""
    stats = []
    
    if companies_df is not None:
        stats.append(f"Companies Dataset: {len(companies_df)} records")
        if "is_fraud" in companies_df.columns:
            fraud_count = companies_df["is_fraud"].sum()
            stats.append(f"Fraud Companies: {fraud_count} ({fraud_count/len(companies_df)*100:.2f}%)")
        if "turnover" in companies_df.columns:
            stats.append(f"Total Turnover: ₹{companies_df['turnover'].sum():,.0f}")
            stats.append(f"Average Turnover: ₹{companies_df['turnover'].mean():,.0f}")
        if "location" in companies_df.columns:
            stats.append(f"Locations Covered: {companies_df['location'].nunique()}")
            top_locations = companies_df["location"].value_counts().head(3)
            stats.append(f"Top 3 Locations: {dict(top_locations)}")
    
    if invoices_df is not None:
        stats.append(f"Invoices Dataset: {len(invoices_df)} records")
        if "amount" in invoices_df.columns:
            stats.append(f"Total Invoice Value: ₹{invoices_df['amount'].sum():,.0f}")
            stats.append(f"Average Invoice Value: ₹{invoices_df['amount'].mean():,.0f}")
        if "itc_claimed" in invoices_df.columns:
            stats.append(f"Total ITC Claims: ₹{invoices_df['itc_claimed'].sum():,.0f}")
            stats.append(f"Average ITC per Invoice: ₹{invoices_df['itc_claimed'].mean():,.0f}")
    
    return "\n".join(stats)

# Function to get fraud insights
def get_fraud_insights():
    """Get insights about fraud patterns"""
    insights = []
    
    if companies_df is not None and "is_fraud" in companies_df.columns:
        fraud_companies = companies_df[companies_df["is_fraud"] == 1]
        if len(fraud_companies) > 0:
            insights.append(f"Fraud Analysis: {len(fraud_companies)} fraudulent companies identified")
            
            if "turnover" in fraud_companies.columns:
                avg_fraud_turnover = fraud_companies["turnover"].mean()
                insights.append(f"Average turnover of fraud companies: ₹{avg_fraud_turnover:,.0f}")
                
            if "location" in fraud_companies.columns:
                fraud_by_location = fraud_companies["location"].value_counts().head(5)
                insights.append(f"Top fraud locations: {dict(fraud_by_location)}")
    
    return "\n".join(insights)

# Enhanced context for the LLM
def get_enhanced_context():
    """Get enhanced context including fraud insights and data statistics"""
    context = "You are a GST tax compliance and fraud detection expert assistant. "
    context += "You have access to a dataset of companies and their invoices. "
    context += "Provide accurate, data-driven responses based on the following information:\n\n"
    
    context += "=== DATASET STATISTICS ===\n"
    context += get_data_statistics()
    context += "\n\n"
    
    context += "=== FRAUD INSIGHTS ===\n"
    context += get_fraud_insights()
    context += "\n\n"
    
    context += "=== MODEL CAPABILITIES ===\n"
    context += f"Graph Neural Network Model: {'Trained' if model_info['model_trained'] else 'Not trained'}\n"
    context += f"Network Analysis: {'Available' if model_info['networkx_exists'] else 'Limited'}\n"
    
    return context

# Function to get chatbot response
def get_chatbot_response(question):
    """Get response from Groq LLM with tax data context"""
    # Enhanced system context
    system_context = get_enhanced_context()
    
    # Prepare messages with context
    messages = [
        {"role": "system", "content": system_context},
    ]
    
    # Add recent conversation history (last 5 messages)
    if "messages" in st.session_state:
        messages.extend(st.session_state.messages[-10:])  # Last 5 exchanges (10 messages)
    
    # Add current user question
    messages.append({"role": "user", "content": question})
    
    try:
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a question about GST companies, invoices, or fraud detection...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing GST data..."):
            # Get response from chatbot
            ai_response = get_chatbot_response(user_input)
            
            # Display response
            st.markdown(ai_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Add quick action buttons for common queries
st.divider()
st.subheader("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Top Fraud Companies"):
        quick_query = "Show me the top 5 companies with the highest fraud probability"
        st.session_state.messages.append({"role": "user", "content": quick_query})
        with st.chat_message("user"):
            st.markdown(quick_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_chatbot_response(quick_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    if st.button("💰 High Value Invoices"):
        quick_query = "List invoices above ₹1,00,000 sorted by value"
        st.session_state.messages.append({"role": "user", "content": quick_query})
        with st.chat_message("user"):
            st.markdown(quick_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_chatbot_response(quick_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with col3:
    if st.button("📍 Fraud by Location"):
        quick_query = "Show fraud distribution by location/state"
        st.session_state.messages.append({"role": "user", "content": quick_query})
        with st.chat_message("user"):
            st.markdown(quick_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_chatbot_response(quick_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with col4:
    if st.button("📈 ITC Analysis"):
        quick_query = "Which companies have claimed the most ITC? Show top 5"
        st.session_state.messages.append({"role": "user", "content": quick_query})
        with st.chat_message("user"):
            st.markdown(quick_query)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_chatbot_response(quick_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})