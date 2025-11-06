import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="VIRAHUB Prototype", layout="wide")
st.title("VIRAHUB — Partner Matching Prototype")

# --- Sample Data ---
businesses = [
    {"id": 1, "name": "Bakery A", "industry": "Food", "funding_needed": 5000, "stage": "early"},
    {"id": 2, "name": "Tech Startup B", "industry": "Software", "funding_needed": 20000, "stage": "growth"},
    {"id": 3, "name": "Handicraft C", "industry": "Artisan", "funding_needed": 1000, "stage": "idea"}
]

partners = [
    {"id": 1, "name": "Investor X", "industry_focus": ["Food"], "funding_capacity": 10000, "mentorship_capacity": 0},
    {"id": 2, "name": "Mentor Y", "industry_focus": ["Software"], "funding_capacity": 0, "mentorship_capacity": 1},
    {"id": 3, "name": "Partner Z", "industry_focus": ["Artisan"], "funding_capacity": 2000, "mentorship_capacity": 0}
]

# --- Sidebar controls for scoring ---
st.sidebar.header("Rule Weights")
industry_weight = st.sidebar.slider("Industry Match Weight", 0.0, 1.0, 0.5)
funding_weight = st.sidebar.slider("Funding Match Weight", 0.0, 1.0, 0.3)
mentorship_weight = st.sidebar.slider("Mentorship Match Weight", 0.0, 1.0, 0.2)
top_n = st.sidebar.number_input("Top N matches", min_value=1, max_value=len(partners), value=3)

# --- Matching function ---
def match_score(biz, partner):
    score = 0
    # Industry match
    if biz["industry"] in partner["industry_focus"]:
        score += industry_weight
    # Funding match
    if partner["funding_capacity"] >= biz["funding_needed"]:
        score += funding_weight
    # Mentorship match
    if partner["mentorship_capacity"] > 0 and biz["stage"] != "growth":
        score += mentorship_weight
    return score

# --- Select a business ---
biz_options = [f'{b["name"]} ({b["industry"]})' for b in businesses]
selected_idx = st.selectbox("Select a business:", options=list(range(len(businesses))), format_func=lambda i: biz_options[i])
selected_biz = businesses[selected_idx]
st.subheader(f"Selected: {selected_biz['name']}")

# --- Compute matches ---
matches = []
for p in partners:
    score = match_score(selected_biz, p)
    matches.append({"partner": p, "score": score})

# --- Display top matches ---
matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:top_n]
df = pd.DataFrame([{
    "Partner Name": m["partner"]["name"],
    "Industries": ", ".join(m["partner"]["industry_focus"]),
    "Funding Capacity": m["partner"]["funding_capacity"],
    "Mentorship Capacity": m["partner"]["mentorship_capacity"],
    "Score": m["score"]
} for m in matches])
st.dataframe(df)

# --- Optional: Bar chart ---
st.subheader("Match Scores")
st.bar_chart(df.set_index("Partner Name")["Score"])
