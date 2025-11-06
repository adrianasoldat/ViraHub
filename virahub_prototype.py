import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Synthetic Data Generator
# -----------------------------
INDUSTRIES = ["Fintech","Health","EdTech","Agritech","E-commerce","AI/ML","CleanTech","Logistics"]
STAGES = ["idea","seed","growth","scale"]

def random_business(i):
    return {
        "id": f"biz_{i}",
        "name": f"Business {i}",
        "industry": random.choice(INDUSTRIES),
        "stage": random.choice(STAGES),
        "funding_needed": random.randint(50_000,500_000),
        "mentorship_needed": round(random.uniform(1,10),1),
        "location": random.choice(["Local","Regional","Remote"]),
        "tags": random.sample(["AI","FinOps","Growth","Marketing","SaaS"], 2)
    }

def random_partner(i):
    return {
        "id": f"prt_{i}",
        "name": f"Partner {i}",
        "industry_focus": random.sample(INDUSTRIES, random.randint(1,2)),
        "funding_capacity": random.randint(100_000,600_000),
        "mentorship_capacity": round(random.uniform(1,10),1),
        "preferred_stages": random.sample(STAGES, random.randint(1,2)),
        "location": random.choice(["Local","Regional","Remote"]),
        "tags": random.sample(["AI","FinOps","Growth","Marketing","SaaS"],3)
    }

def pair_to_features(biz, prt):
    return {
        "industry_match": 1 if biz["industry"] in prt["industry_focus"] else 0,
        "funding_coverage": min(1.0, prt["funding_capacity"]/ (biz["funding_needed"]+1)),
        "mentorship_match": max(0.0, 1 - abs(biz["mentorship_needed"]-prt["mentorship_capacity"])/10),
        "stage_compat": 1 if biz["stage"] in prt["preferred_stages"] else 0,
        "location_same": 1 if biz["location"] == prt["location"] else 0,
        "tag_overlap": len(set(biz["tags"]) & set(prt["tags"]))/max(len(biz["tags"]),1)
    }

def features_dicts_to_matrix(feature_dicts):
    keys = ["industry_match","funding_coverage","mentorship_match","stage_compat","location_same","tag_overlap"]
    X = np.array([[f[k] for k in keys] for f in feature_dicts], dtype=float)
    return X, keys

# -----------------------------
# Rule-based matching
# -----------------------------
DEFAULT_WEIGHTS = {
    "industry_match":0.35, "funding_coverage":0.25, "mentorship_match":0.2,
    "stage_compat":0.1, "location_same":0.05, "tag_overlap":0.05
}

def compute_rule_score(features, weights=DEFAULT_WEIGHTS):
    total = sum(weights.values())
    normalized = {k:v/total for k,v in weights.items()}
    return sum(normalized[k]*features.get(k,0) for k in normalized)

# -----------------------------
# ML Model
# -----------------------------
class MLCompatibilityModel:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(32,16), max_iter=300, random_state=42))
        ])
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        preds = self.model.predict(X)
        return np.clip(preds,0.0,1.0)

# -----------------------------
# Find top matches
# -----------------------------
def find_top_matches(biz, partners, ml_model=None, ml_weight=0.5, top_n=5):
    results=[]
    # Build ML features if model exists
    if ml_model:
        X_dicts = [pair_to_features(biz,p) for p in partners]
        X, keys = features_dicts_to_matrix(X_dicts)
        ml_scores = ml_model.predict(X)
    else:
        ml_scores = [None]*len(partners)

    for p, ml_score in zip(partners, ml_scores):
        feats = pair_to_features(biz,p)
        rule_score = compute_rule_score(feats)
        combined = rule_score if ml_score is None else (1-ml_weight)*rule_score + ml_weight*ml_score
        results.append({
            "partner": p,
            "features": feats,
            "rule_score": round(rule_score,3),
            "ml_score": round(float(ml_score),3) if ml_score is not None else None,
            "combined_score": round(combined,3)
        })
    return sorted(results, key=lambda r:r["combined_score"], reverse=True)[:top_n]

# -----------------------------
# Streamlit App
# -----------------------------
st.title("VIRAHUB Partner Matching (Simplified + ML)")

# Generate data
random.seed(42)
businesses = [random_business(i) for i in range(5)]
partners = [random_partner(i) for i in range(15)]

# Sidebar: ML weight
ml_weight = st.sidebar.slider("ML weight (combined score)", 0.0, 1.0, 0.5)

# Sidebar: train small ML model
train_ml = st.sidebar.button("Train ML model")
ml_model = MLCompatibilityModel()

# Prepare training data from all pairs
training_pairs = []
for b in businesses:
    for p in partners:
        feats = pair_to_features(b,p)
        # hidden "ground truth" score: simple rule + noise
        label = compute_rule_score(feats) + np.random.normal(0,0.05)
        label = np.clip(label,0,1)
        training_pairs.append({"features":feats,"label":label})

if train_ml:
    X_dicts = [t["features"] for t in training_pairs]
    X, keys = features_dicts_to_matrix(X_dicts)
    y = np.array([t["label"] for t in training_pairs])
    ml_model.fit(X, y)
    st.success("ML model trained!")

# Select business
biz_names = [f"{b['name']} ({b['industry']})" for b in businesses]
selected_idx = st.selectbox("Select a business:", range(len(businesses)), format_func=lambda i: biz_names[i])
selected_biz = businesses[selected_idx]

# Show top matches
matches = find_top_matches(selected_biz, partners, ml_model if train_ml else None, ml_weight=ml_weight, top_n=5)

df = pd.DataFrame([{
    "Partner": m["partner"]["name"],
    "Industry Focus": ", ".join(m["partner"]["industry_focus"]),
    "Rule Score": m["rule_score"],
    "ML Score": m["ml_score"] if m["ml_score"] is not None else "—",
    "Combined Score": m["combined_score"]
} for m in matches])

st.subheader("Top Partner Matches")
st.dataframe(df)
