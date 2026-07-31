import streamlit as st
import joblib
import pandas as pd
from feature_columns import FEATURE_COLUMNS
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.write("""
### AI Startup Success Predictor

**Machine Learning Model**
- Random Forest Classifier

**Best Accuracy**
- 82.16%

**Dataset**
- Startup Success Prediction Dataset

**Developed Using**
- Python
- Scikit-Learn
- Streamlit
""")

# -----------------------------
# Main Page
# -----------------------------
st.title("🚀 AI Startup Success Predictor")

st.markdown("""
Welcome to the **AI Startup Success Predictor**.

This application predicts whether a startup is likely to succeed based on its business details using a trained **Random Forest Machine Learning Model**.
""")

st.divider()

st.subheader("📋 Enter Startup Information")

# -----------------------------
# Basic Startup Information
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    latitude = st.number_input(
        "Latitude",
        value=37.7749,
        format="%.4f"
    )

    age_first_funding_year = st.number_input(
        "Age at First Funding (Years)",
        min_value=0.0,
        value=1.0
    )

    age_last_funding_year = st.number_input(
        "Age at Last Funding (Years)",
        min_value=0.0,
        value=2.0
    )

    funding_total_usd = st.number_input(
        "Total Funding (USD)",
        min_value=0.0,
        value=1000000.0
    )

    funding_rounds = st.number_input(
        "Funding Rounds",
        min_value=0,
        value=2
    )

    milestones = st.number_input(
        "Milestones",
        min_value=0,
        value=2
    )


with col2:

    longitude = st.number_input(
        "Longitude",
        value=-122.4194,
        format="%.4f"
    )

    age_first_milestone_year = st.number_input(
        "Age at First Milestone",
        min_value=0.0,
        value=1.0
    )

    age_last_milestone_year = st.number_input(
        "Age at Last Milestone",
        min_value=0.0,
        value=3.0
    )

    relationships = st.number_input(
        "Relationships",
        min_value=0,
        value=5
    )

    avg_participants = st.number_input(
        "Average Participants",
        min_value=1,
        value=2
    )

    founded_year = st.number_input(
        "Founded Year",
        min_value=1980,
        max_value=2035,
        value=2015
    )

st.divider()

st.subheader("Investment Information")

col3, col4 = st.columns(2)

with col3:

    has_VC = st.selectbox(
        "Has Venture Capital?",
        [0, 1]
    )

    has_angel = st.selectbox(
        "Has Angel Investment?",
        [0, 1]
    )

    has_roundA = st.selectbox(
        "Has Round A?",
        [0, 1]
    )

    has_roundB = st.selectbox(
        "Has Round B?",
        [0, 1]
    )


with col4:

    has_roundC = st.selectbox(
        "Has Round C?",
        [0, 1]
    )

    has_roundD = st.selectbox(
        "Has Round D?",
        [0, 1]
    )

    is_top500 = st.selectbox(
        "Top 500 Startup?",
        [0, 1]
    )

st.divider()

st.subheader("Category & Location")

state = st.selectbox(
    "State",
    [
        "CA",
        "NY",
        "TX",
        "MA",
        "FL",
        "WA",
        "IL",
        "CO",
        "PA",
        "Other"
    ]
)

category = st.selectbox(
    "Category",
    [
        "software",
        "web",
        "mobile",
        "enterprise",
        "advertising",
        "games_video",
        "ecommerce",
        "biotech",
        "consulting",
        "analytics",
        "education",
        "finance",
        "health",
        "hardware",
        "security",
        "travel",
        "social",
        "other"
    ]
)


import pandas as pd

if st.button("Predict Startup Success"):

    first_funding_year = founded_year + age_first_funding_year
    last_funding_year = founded_year + age_last_funding_year

    # Create all 104 features with default value 0
    input_df = pd.DataFrame(
    [[0.0] * len(FEATURE_COLUMNS)],
    columns=FEATURE_COLUMNS
)

    # Numerical features
    input_df.at[0, "latitude"] = latitude
    input_df.at[0, "longitude"] = longitude
    input_df.at[0, "age_first_funding_year"] = age_first_funding_year
    input_df.at[0, "age_last_funding_year"] = age_last_funding_year
    input_df.at[0, "age_first_milestone_year"] = age_first_milestone_year
    input_df.at[0, "age_last_milestone_year"] = age_last_milestone_year
    input_df.at[0, "relationships"] = relationships
    input_df.at[0, "funding_rounds"] = funding_rounds
    input_df.at[0, "funding_total_usd"] = funding_total_usd
    input_df.at[0, "milestones"] = milestones
    input_df.at[0, "has_VC"] = has_VC
    input_df.at[0, "has_angel"] = has_angel
    input_df.at[0, "has_roundA"] = has_roundA
    input_df.at[0, "has_roundB"] = has_roundB
    input_df.at[0, "has_roundC"] = has_roundC
    input_df.at[0, "has_roundD"] = has_roundD
    input_df.at[0, "avg_participants"] = avg_participants
    input_df.at[0, "is_top500"] = is_top500
    input_df.at[0, "founded_year"] = founded_year
    input_df.at[0, "first_funding_year"] = first_funding_year
    input_df.at[0, "last_funding_year"] = last_funding_year

    # State flags
    if state == "CA":
        input_df.at[0, "is_CA"] = 1
        input_df.at[0, "state_code_CA"] = 1
    elif state == "NY":
        input_df.at[0, "is_NY"] = 1
        input_df.at[0, "state_code_NY"] = 1
    elif state == "TX":
        input_df.at[0, "is_TX"] = 1
        input_df.at[0, "state_code_TX"] = 1
    elif state == "MA":
        input_df.at[0, "is_MA"] = 1
        input_df.at[0, "state_code_MA"] = 1
    elif state == "FL":
        input_df.at[0, "state_code_FL"] = 1
    elif state == "WA":
        input_df.at[0, "state_code_WA"] = 1
    elif state == "IL":
        input_df.at[0, "state_code_IL"] = 1
    elif state == "CO":
        input_df.at[0, "state_code_CO"] = 1
    elif state == "PA":
        input_df.at[0, "state_code_PA"] = 1
    else:
        input_df.at[0, "is_otherstate"] = 1

    # Category flags
    category_map = {
        "software": ("is_software", "category_code_software"),
        "web": ("is_web", "category_code_web"),
        "mobile": ("is_mobile", "category_code_mobile"),
        "enterprise": ("is_enterprise", "category_code_enterprise"),
        "advertising": ("is_advertising", None),
        "games_video": ("is_gamesvideo", "category_code_games_video"),
        "ecommerce": ("is_ecommerce", "category_code_ecommerce"),
        "biotech": ("is_biotech", "category_code_biotech"),
        "consulting": ("is_consulting", "category_code_consulting"),
        "analytics": (None, "category_code_analytics"),
        "education": (None, "category_code_education"),
        "finance": (None, "category_code_finance"),
        "health": (None, "category_code_health"),
        "hardware": (None, "category_code_hardware"),
        "security": (None, "category_code_security"),
        "travel": (None, "category_code_travel"),
        "social": (None, "category_code_social"),
        "other": ("is_othercategory", "category_code_other"),
    }

    if category in category_map:
        flag1, flag2 = category_map[category]
        if flag1:
            input_df.at[0, flag1] = 1
        if flag2:
            input_df.at[0, flag2] = 1

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Startup is likely to Succeed")
    else:
        st.error("❌ Startup is likely to Fail")