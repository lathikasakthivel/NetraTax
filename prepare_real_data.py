"""
Tax Fraud Detection Model Training
Using Real Companies and Invoices Datasets
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("\n" + "=" * 80)
print("TAX FRAUD DETECTION - MODEL TRAINING WITH REAL DATASETS")
print("=" * 80)

# ============================================================================
# 1. LOAD REAL DATASETS
# ============================================================================

print("\n📂 Loading datasets...")
companies = pd.read_csv("c:\\BIG HACK\\companies.csv")
invoices = pd.read_csv("c:\\BIG HACK\\invoices.csv")

print(f"✅ Companies: {companies.shape[0]} rows, {companies.shape[1]} columns")
print(f"✅ Invoices: {invoices.shape[0]} rows, {invoices.shape[1]} columns")

# ============================================================================
# 2. DATASET ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("📊 DATASET ANALYSIS")
print("=" * 80)

print(f"\n🚨 Fraud Distribution in Companies:")
fraud_dist = companies['is_fraud'].value_counts()
fraud_pct = (fraud_dist / len(companies) * 100).round(2)
print(f"   Non-Fraudulent: {fraud_dist.get(0, 0)} ({fraud_pct.get(0, 0)}%)")
print(f"   Fraudulent: {fraud_dist.get(1, 0)} ({fraud_pct.get(1, 0)}%)")

print(f"\n📍 Locations: {companies['location'].nunique()} unique locations")
print(companies['location'].value_counts().to_string())

print(f"\n💰 Turnover Statistics:")
print(f"   Min: ₹{companies['turnover'].min():,.2f}")
print(f"   Max: ₹{companies['turnover'].max():,.2f}")
print(f"   Mean: ₹{companies['turnover'].mean():,.2f}")
print(f"   Median: ₹{companies['turnover'].median():,.2f}")

invoices['date'] = pd.to_datetime(invoices['date'])
print(f"\n📅 Invoices Date Range:")
print(f"   From: {invoices['date'].min()}")
print(f"   To: {invoices['date'].max()}")

itc_dist = invoices['itc_claimed'].value_counts()
print(f"\n🏛️ ITC Claimed Distribution:")
print(f"   Not Claimed: {itc_dist.get(0, 0)} ({itc_dist.get(0, 0)/len(invoices)*100:.1f}%)")
print(f"   Claimed: {itc_dist.get(1, 0)} ({itc_dist.get(1, 0)/len(invoices)*100:.1f}%)")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("🔄 DATA PREPROCESSING")
print("=" * 80)

# Save processed data
output_dir = Path(__file__).parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

companies.to_csv(output_dir / "companies_processed.csv", index=False)
invoices.to_csv(output_dir / "invoices_processed.csv", index=False)

print(f"\n✅ Companies saved: {output_dir / 'companies_processed.csv'}")
print(f"✅ Invoices saved: {output_dir / 'invoices_processed.csv'}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("⚙️  FEATURE ENGINEERING")
print("=" * 80)

# Analyze companies based on invoice patterns
company_features = []

for company_id in companies['company_id'].unique():
    company_data = companies[companies['company_id'] == company_id].iloc[0]
    
    # Count sent and received invoices
    sent_invoices = invoices[invoices['seller_id'] == company_id]
    received_invoices = invoices[invoices['buyer_id'] == company_id]
    
    features = {
        'company_id': company_id,
        'turnover': company_data['turnover'],
        'sent_invoices': len(sent_invoices),
        'received_invoices': len(received_invoices),
        'total_sent_amount': sent_invoices['amount'].sum() if len(sent_invoices) > 0 else 0,
        'total_received_amount': received_invoices['amount'].sum() if len(received_invoices) > 0 else 0,
        'avg_sent_amount': sent_invoices['amount'].mean() if len(sent_invoices) > 0 else 0,
        'avg_received_amount': received_invoices['amount'].mean() if len(received_invoices) > 0 else 0,
        'itc_claimed_sent': sent_invoices['itc_claimed'].sum() if len(sent_invoices) > 0 else 0,
        'itc_claimed_received': received_invoices['itc_claimed'].sum() if len(received_invoices) > 0 else 0,
        'is_fraud': company_data['is_fraud'],
        'location': company_data['location']
    }
    
    company_features.append(features)

features_df = pd.DataFrame(company_features)
features_df.to_csv(output_dir / "company_features.csv", index=False)

print(f"\n✅ Generated {len(features_df)} company feature vectors")
print(f"Features: {list(features_df.columns)}")
print(f"✅ Saved to: {output_dir / 'company_features.csv'}")

# ============================================================================
# 5. GRAPH STRUCTURE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("📊 GRAPH STRUCTURE")
print("=" * 80)

# Filter invoices to only include seller-buyer pairs in companies dataset
company_ids_set = set(companies['company_id'].unique())
valid_invoices = invoices[
    (invoices['seller_id'].isin(company_ids_set)) & 
    (invoices['buyer_id'].isin(company_ids_set))
].copy()

print(f"\nTotal Companies (Nodes): {len(company_ids_set)}")
print(f"Total Transactions (Edges): {len(valid_invoices)}")
print(f"Network Density: {len(valid_invoices) / (len(company_ids_set) * (len(company_ids_set) - 1)):.6f}")

# Edge distribution
print(f"\nEdge Statistics:")
out_degrees = [len(valid_invoices[valid_invoices['seller_id'] == cid]) for cid in company_ids_set]
print(f"   Avg edges per company: {len(valid_invoices) / len(company_ids_set):.2f}")
print(f"   Max outgoing edges: {max(out_degrees) if out_degrees else 0}")
print(f"   Max incoming edges: {max([len(valid_invoices[valid_invoices['buyer_id'] == cid]) for cid in company_ids_set]) if company_ids_set else 0}")

# ============================================================================
# 6. SUITABILITY VERDICT
# ============================================================================

print("\n" + "=" * 80)
print("✅ SUITABILITY ASSESSMENT")
print("=" * 80)

criteria = {
    'Fraud Labels Present': 'is_fraud' in companies.columns,
    'Seller-Buyer Network': ('seller_id' in invoices.columns and 'buyer_id' in invoices.columns),
    'Sufficient Companies': len(companies) >= 100,
    'Sufficient Transactions': len(invoices) >= 1000,
    'Good Fraud Ratio': 0.05 <= fraud_dist.get(1, 0) / len(companies) <= 0.95,
    'Financial Features': 'turnover' in companies.columns and 'amount' in invoices.columns,
    'ITC Information': 'itc_claimed' in invoices.columns,
    'No Missing Values': companies.isnull().sum().sum() == 0 and invoices.isnull().sum().sum() == 0,
}

print("\nCriteria Assessment:")
for criterion, result in criteria.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {criterion}")

passed = sum(criteria.values())
total = len(criteria)

print(f"\n{'='*80}")
print(f"PASSED: {passed}/{total} criteria")

if passed >= 6:
    print("\n🎉 VERDICT: ✅ HIGHLY SUITABLE FOR GNN FRAUD DETECTION")
    print("✅ Status: READY FOR MODEL TRAINING")
    print("\n📋 Recommendation: Proceed with the following steps:")
    print("   1. Build graph from transactions")
    print("   2. Convert to PyTorch Geometric format")
    print("   3. Train Graph Convolutional Network (GCN)")
    print("   4. Evaluate on test set")
    print("   5. Deploy model via Flask API")
elif passed >= 4:
    print("\n🟡 VERDICT: SUITABLE with minor improvements")
else:
    print("\n⚠️ VERDICT: NOT SUITABLE - Needs significant preprocessing")

print("\n" + "=" * 80)
print("Analysis Complete! Ready for model training.")
print("=" * 80 + "\n")
