from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
IMAGES = ROOT / "images"

st.set_page_config(page_title="FraudLens", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS / "fraud_xgb.joblib")
    features = json.loads((MODELS / "features.json").read_text())
    cat_map = json.loads((MODELS / "category_map.json").read_text())
    threshold = json.loads((MODELS / "threshold.json").read_text())["threshold"]
    explainer = shap.TreeExplainer(model)
    return model, features, cat_map, threshold, explainer


model, FEATURES, CAT_MAP, THRESHOLD, explainer = load_artifacts()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LABELS = {
    "entertainment": "Entertainment", "food_dining": "Restaurants", "gas_transport": "Fuel & transport",
    "grocery_net": "Groceries (online)", "grocery_pos": "Groceries (in store)", "health_fitness": "Health & fitness",
    "home": "Home", "kids_pets": "Kids & pets", "misc_net": "Misc. (online)", "misc_pos": "Misc. (in store)",
    "personal_care": "Personal care", "shopping_net": "Shopping (online)", "shopping_pos": "Shopping (in store)",
    "travel": "Travel",
}
EXAMPLES = {
    "Normal grocery shop": dict(amt=45, usual=60, category="grocery_pos", hour=14, day=5, age=45,
                                distance=70, txn_24h=3, hours_since=8.0, city_pop=20000),
    "Late-night online purchase": dict(amt=950, usual=60, category="shopping_net", hour=23, day=2, age=45,
                                       distance=80, txn_24h=6, hours_since=0.5, city_pop=20000),
    "Small payment, card-testing pattern": dict(amt=9, usual=70, category="misc_net", hour=2, day=1, age=52,
                                                distance=75, txn_24h=8, hours_since=0.2, city_pop=5000),
}

st.title("FraudLens")
st.caption("Card fraud detection · XGBoost trained on 1.3M transactions · PR-AUC 0.91 on unseen future data")

tab_score, tab_perf, tab_patterns = st.tabs(["Score a transaction", "Model performance", "Fraud patterns"])

# ---------- Tab 1: score a transaction ----------
with tab_score:
    ex_name = st.selectbox("Start from an example", list(EXAMPLES))
    ex = EXAMPLES[ex_name]
    left, right = st.columns([1, 2], gap="large")

    with left:
        k = ex_name  # keys reset the inputs when the example changes
        amt = st.number_input("Amount ($)", min_value=1.0, value=float(ex["amt"]), key=f"amt{k}")
        usual = st.number_input("Customer's usual spend ($)", min_value=1.0, value=float(ex["usual"]), key=f"usual{k}")
        cats = list(CAT_MAP)
        category = st.selectbox("Shop category", cats, index=cats.index(ex["category"]),
                                format_func=lambda c: LABELS.get(c, c), key=f"cat{k}")
        hour = st.slider("Hour of day", 0, 23, ex["hour"], key=f"hour{k}")
        day = st.selectbox("Day of week", DAYS, index=ex["day"], key=f"day{k}")
        age = st.slider("Customer age", 18, 95, ex["age"], key=f"age{k}")
        distance = st.number_input("Distance from home (km)", min_value=0.0, value=float(ex["distance"]), key=f"dist{k}")
        txn_24h = st.number_input("Payments in previous 24 hours", min_value=0, value=ex["txn_24h"], step=1, key=f"txn{k}")
        hours_since = st.number_input("Hours since previous payment", min_value=0.0, max_value=168.0,
                                      value=float(ex["hours_since"]), key=f"since{k}")
        city_pop = st.number_input("Population of customer's city", min_value=100, value=ex["city_pop"],
                                   step=1000, key=f"pop{k}")

    row = pd.DataFrame([{
        "amt": amt,
        "amt_ratio": amt / usual,
        "hour": hour,
        "is_night": int(hour in (22, 23, 0, 1, 2, 3)),
        "day_of_week": DAYS.index(day),
        "age": age,
        "distance_km": distance,
        "txn_24h": txn_24h,
        "hours_since_last": hours_since,
        "city_pop": city_pop,
        "category_code": CAT_MAP[category],
    }])[FEATURES]

    p = float(model.predict_proba(row)[0, 1])
    score = round(p * 100)

    with right:
        if p < THRESHOLD:
            decision, colour = "Approve", "green"
        elif p < 0.9:
            decision, colour = "Send for review", "orange"
        else:
            decision, colour = "Block and contact customer", "red"

        c1, c2 = st.columns(2)
        c1.metric("Fraud risk score", f"{score} / 100")
        c2.markdown(f"### :{colour}[{decision}]")
        st.progress(min(score, 100))
        st.caption(f"Payments above {THRESHOLD:.2f} are flagged. This threshold was chosen to minimise "
                   "total cost (missed fraud + $8 review per alert).")

        st.subheader("Why the model thinks so")
        st.caption("Red bars push the risk up, blue bars pull it down (model's internal log-odds score).")
        sv = explainer(row)
        shap.plots.waterfall(sv[0], max_display=8, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)


def show_image(name, caption):
    path = IMAGES / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: images/{name}")


# ---------- Tab 2: model performance ----------
with tab_perf:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", "0.91", "vs 0.17 baseline")
    m2.metric("Fraud caught", "93.4%")
    m3.metric("False alarms", "3.7 per 1,000")
    m4.metric("Fraud cost reduction", "95%", "$1.13M → $52.8k")

    c1, c2 = st.columns(2)
    with c1:
        show_image("threshold_cost.png", "Choosing the alert threshold by cost")
        show_image("confusion_matrix.png", "Confusion matrix on the test period")
    with c2:
        show_image("pr_curve.png", "Precision–recall: XGBoost vs logistic regression")
        show_image("shap_beeswarm.png", "How each feature affects fraud risk")

# ---------- Tab 3: fraud patterns ----------
with tab_patterns:
    c1, c2 = st.columns(2)
    with c1:
        show_image("fraud_by_hour.png", "Fraud rate by hour of day")
        show_image("fraud_by_amt_ratio.png", "Fraud rate vs amount compared with usual spend")
    with c2:
        show_image("fraud_by_category.png", "Fraud rate by shop category")
        show_image("amount_distribution.png", "Amounts: fraud vs genuine")

st.divider()
st.caption("Built by Shraddha Tole · Data: Sparkov synthetic card transactions (Kaggle) · "
           "Portfolio project, not a production system.")