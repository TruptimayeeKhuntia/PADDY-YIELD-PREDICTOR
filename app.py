"""
app.py
------
Paddy Yield Predictor - Streamlit Application

Run:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "paddydataset.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "paddy_yield_predictor.joblib"
)

TARGET = "Paddy yield(in Kg)"


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Paddy Yield Predictor",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# 3. FEATURE EXPLANATIONS
# ============================================================

FEATURE_HELP = {

    "Hectares":
        "Area of land used for paddy cultivation in hectares.",

    "Agriblock":
        "Agricultural block or area where the crop is cultivated.",

    "Variety":
        "Type or variety of paddy cultivated.",

    "Soil Types":
        "Type of soil present in the field.",

    "Seedrate(in Kg)":
        "Amount of seed used for cultivation in kilograms.",

    "LP_Mainfield(in Tonnes)":
        "Main field land-preparation input measured in tonnes.",

    "Nursery":
        "Information related to nursery cultivation.",

    "Nursery area (Cents)":
        "Area used for nursery cultivation in cents.",

    "LP_nurseryarea(in Tonnes)":
        "Land-preparation input for nursery area.",

    "DAP_20days":
        "DAP fertilizer applied around 20 days.",

    "Weed28D_thiobencarb":
        "Weed-control treatment using Thiobencarb.",

    "Urea_40Days":
        "Urea fertilizer applied around 40 days.",

    "Potassh_50Days":
        "Potassium fertilizer applied around 50 days.",

    "Micronutrients_70Days":
        "Micronutrients applied around 70 days.",

    "Pest_60Day":
        "Pest-control treatment applied around 60 days.",

    "Rain measures":
        "Rainfall-related measurement during crop growth.",

    "AI measures":
        "Agricultural/environmental measurement used by the model.",

    "Min/Max temp ranges":
        "Minimum and maximum temperature measurements.",

    "Inst Wind Speed_D1_D30(in Knots)":
        "Wind speed during days 1 to 30 in knots.",

    "Wind Direction_D1_D30":
        "Wind direction recorded during days 1 to 30."
}


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Model file was not found.

Expected location:
{MODEL_PATH}

Please make sure this file exists:

models/paddy_yield_predictor.joblib
"""
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# 5. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"""
Dataset file was not found.

Expected location:
{DATA_PATH}
"""
        )

    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    return df


# ============================================================
# 6. LOAD RESOURCES
# ============================================================

try:

    model = load_model()
    df = load_dataset()

except Exception as error:

    st.error("❌ Unable to load project files.")

    st.exception(error)

    st.stop()


# ============================================================
# 7. CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    st.error(
        f"""
❌ Target column was not found.

Expected:
{TARGET}

Available columns:
{list(df.columns)}
"""
    )

    st.stop()


# ============================================================
# 8. FEATURES
# ============================================================

X = df.drop(
    columns=[TARGET]
)


# ============================================================
# 9. HERO SECTION - HTML ONLY
# ============================================================

st.title("🌾 Paddy Yield Predictor")

st.subheader("AI-Powered Paddy Yield Prediction System")

st.write(
    "Predict paddy yield using agricultural, "
    "environmental and crop-management factors."
)

st.info(
    "🤖 Machine Learning  |  🌱 Smart Agriculture  |  📊 Data Driven"
)

st.divider()


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌾 Paddy Yield Predictor")

    st.divider()

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.subheader("📌 About Project")

    st.write(
        """
        **Paddy Yield Predictor** is an AI and Machine
        Learning based application that predicts paddy
        yield using agricultural, environmental and
        crop-management factors.
        """
    )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Objective")

    st.markdown(
        """
        - 🌾 Predict paddy yield in kilograms
        - 📊 Analyze factors affecting crop production
        - 🌱 Understand agricultural inputs
        - 🤖 Support data-driven agricultural decisions
        """
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Model")

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Learning Type:** Supervised Learning")

    st.write("**Problem Type:** Regression")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("📊 Dataset Information")

    st.write(
        """
        The dataset contains agricultural and
        environmental information used for
        predicting paddy yield.
        """
    )

    st.write("**Target Variable:**")

    st.code(
        TARGET,
        language="text"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("🌱 Features Used")

    st.markdown(
        """
        **Crop & Land**

        - Hectares
        - Agriblock
        - Variety
        - Soil Types

        **Seed & Nursery**

        - Seedrate(in Kg)
        - Nursery
        - Nursery area (Cents)
        - LP_Mainfield(in Tonnes)
        - LP_nurseryarea(in Tonnes)

        **Fertilizer & Management**

        - DAP_20days
        - Weed28D_thiobencarb
        - Urea_40Days
        - Potassh_50Days
        - Micronutrients_70Days
        - Pest_60Day

        **Environmental**

        - Rain measures
        - AI measures
        - Min/Max temp ranges

        **Wind**

        - Inst Wind Speed_D1_D30(in Knots)
        - Wind Direction_D1_D30
        """
    )

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    st.subheader("📋 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Features",
            X.shape[1]
        )

    with col2:

        st.metric(
            "Records",
            f"{len(df):,}"
        )

    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.subheader("🛠 Technology Stack")

    st.markdown(
        """
        **Programming**

        🐍 Python

        **Libraries**

        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit

        **Machine Learning**

        - Random Forest Regressor
        - Data Preprocessing
        - Feature Encoding
        - Missing Value Handling

        **Deployment**

        🚀 Streamlit
        """
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📈 Model Performance")

    st.write(
        """
        The model is evaluated using regression
        performance metrics.
        """
    )

    st.markdown(
        """
        - **MAE** — Mean Absolute Error
        - **MSE** — Mean Squared Error
        - **RMSE** — Root Mean Squared Error
        - **R² Score** — Coefficient of Determination
        """
    )

    st.info(
        "Performance values can be added after model evaluation."
    )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 ML Workflow")

    st.markdown(
        """
        1. 📥 Data Collection
        2. 🔍 Data Understanding
        3. 📊 Exploratory Data Analysis
        4. 🧹 Data Cleaning
        5. ⚙️ Data Preprocessing
        6. ✂️ Train-Test Split
        7. 🤖 Model Training
        8. 📈 Model Evaluation
        9. 💾 Model Saving
        10. 🚀 Deployment
        """
    )

    # --------------------------------------------------------
    # PROJECT TYPE
    # --------------------------------------------------------

    st.subheader("🎓 Project Type")

    st.write(
        """
        **AI & Machine Learning Project**

        🌾 Paddy Yield Prediction System
        """ 
    )


# ============================================================
# 11. MAIN INPUT SECTION
# ============================================================

st.header("🌱 Enter Paddy Information")

st.write(
    """
    Enter the agricultural and environmental information
    required to predict the expected paddy yield.
    """
)

st.divider()


# ============================================================
# 12. USER INPUT
# ============================================================

user_input = {}

input_columns = st.columns(3)


for index, column in enumerate(X.columns):

    with input_columns[index % 3]:

        st.subheader(f"🌱 {column}")

        description = FEATURE_HELP.get(
            column,
            "Enter the value for this agricultural feature."
        )

        st.caption(description)

        # ----------------------------------------------------
        # NUMERICAL INPUT
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            median = X[column].median()

            if pd.isna(median):

                median = 0.0

            user_input[column] = st.number_input(
                "Enter value",
                value=float(median),
                key=f"num_{index}"
            )

        # ----------------------------------------------------
        # CATEGORICAL INPUT
        # ----------------------------------------------------

        else:

            values = (
                X[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            values = sorted(values)

            if len(values) > 0:

                user_input[column] = st.selectbox(
                    "Select value",
                    values,
                    key=f"cat_{index}"
                )

            else:

                user_input[column] = st.text_input(
                    "Enter value",
                    key=f"text_{index}"
                )


# ============================================================
# 13. PREDICTION BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🌾 Predict Paddy Yield",
    type="primary",
    use_container_width=True
)


# ============================================================
# 14. PREDICTION
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [user_input]
        )

        # ----------------------------------------------------
        # SAME COLUMN ORDER AS TRAINING
        # ----------------------------------------------------

        input_df = input_df[
            X.columns
        ]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )

        raw_prediction = prediction[0]

        predicted_yield = float(
            raw_prediction
        )

        # ----------------------------------------------------
        # AVOID NEGATIVE YIELD
        # ----------------------------------------------------

        predicted_yield = max(
            0,
            predicted_yield
        )

        # ----------------------------------------------------
        # KG TO TONNES
        # ----------------------------------------------------

        predicted_tonnes = (
            predicted_yield / 1000
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("🌾 Prediction Result")

        st.success(
            "✅ Prediction generated successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🌾 Predicted Paddy Yield",
                f"{predicted_yield:,.2f} Kg"
            )

        with col2:

            st.metric(
                "📦 Estimated Yield",
                f"{predicted_tonnes:,.2f} Tonnes"
            )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.info(
            f"""
            Based on the agricultural and environmental
            information provided, the predicted paddy yield
            is approximately **{predicted_yield:,.2f} Kg**
            or **{predicted_tonnes:,.2f} Tonnes**.
            """
        )

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Entered Information"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(error)


# ============================================================
# 15. FOOTER - HTML ONLY
# ============================================================

st.divider()

st.caption(
    "🌾 Paddy Yield Predictor | "
    "AI & Machine Learning Project | "
    "Smart Agriculture • Data Driven Prediction"
)