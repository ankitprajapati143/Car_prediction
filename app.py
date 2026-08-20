import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction")
st.write("Car Price Prediction using Random Forest Regression")

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    # CSV must be in the same folder as app.py
    df = pd.read_csv("quikr_car.csv")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    return df


df = load_data()

# =========================================================
# DATA CLEANING
# =========================================================

# ---------------------------------------------------------
# kms_driven
# ---------------------------------------------------------

df["kms_driven"] = (
    df["kms_driven"]
    .astype(str)
    .str.replace("kms", "", case=False, regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["kms_driven"] = pd.to_numeric(
    df["kms_driven"],
    errors="coerce"
)

df["kms_driven"] = df["kms_driven"].fillna(
    df["kms_driven"].mean()
)

df["kms_driven"] = df["kms_driven"].astype(int)


# ---------------------------------------------------------
# fuel_type
# ---------------------------------------------------------

df["fuel_type"] = df["fuel_type"].fillna(
    df["fuel_type"].mode()[0]
)


# ---------------------------------------------------------
# year
# ---------------------------------------------------------

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["year"] = df["year"].fillna(
    df["year"].median()
)

df["year"] = df["year"].astype(int)


# ---------------------------------------------------------
# Price
# ---------------------------------------------------------

# Convert Price to numeric.
# "Ask For Price" becomes NaN.

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

# Remove rows where Price is unavailable
df = df.dropna(subset=["Price"])

df["Price"] = df["Price"].astype(int)

# Reset index
df = df.reset_index(drop=True)


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Price", axis=1)

y = df["Price"]


# =========================================================
# CATEGORICAL ENCODING
# =========================================================

# Same method used in your notebook
X = pd.get_dummies(
    X,
    drop_first=True
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


model = train_model(
    X_train,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Model Information")

st.sidebar.write(
    "**Algorithm:** Random Forest Regressor"
)

st.sidebar.write(
    "**Number of Trees:** 100"
)

st.sidebar.write(
    "**Test Size:** 20%"
)

st.sidebar.write(
    "**Random State:** 42"
)

st.sidebar.write(
    f"**R² Score:** {r2:.4f}"
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with col2:
    st.metric(
        "MAE",
        f"₹ {mae:,.0f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"₹ {rmse:,.0f}"
    )

with col4:
    st.metric(
        "Test Data",
        len(X_test)
    )


# =========================================================
# USER INPUT
# =========================================================

st.subheader("🚘 Enter Car Details")


# ---------------------------------------------------------
# Car Name
# ---------------------------------------------------------

car_names = sorted(
    df["name"]
    .dropna()
    .astype(str)
    .unique()
)

car_name = st.selectbox(
    "Car Name",
    car_names
)


# ---------------------------------------------------------
# Company
# ---------------------------------------------------------

companies = sorted(
    df["company"]
    .dropna()
    .astype(str)
    .unique()
)

company = st.selectbox(
    "Company",
    companies
)


# ---------------------------------------------------------
# Year
# ---------------------------------------------------------

min_year = int(df["year"].min())
max_year = int(df["year"].max())

year = st.number_input(
    "Manufacturing Year",
    min_value=min_year,
    max_value=max_year,
    value=max_year,
    step=1
)


# ---------------------------------------------------------
# Kilometers Driven
# ---------------------------------------------------------

min_km = int(df["kms_driven"].min())
max_km = int(df["kms_driven"].max())

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=max_km,
    value=50000,
    step=1000
)


# ---------------------------------------------------------
# Fuel Type
# ---------------------------------------------------------

fuel_types = sorted(
    df["fuel_type"]
    .dropna()
    .astype(str)
    .unique()
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_types
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE USER INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "name": [car_name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })


    # -----------------------------------------------------
    # SAME ENCODING AS TRAINING DATA
    # -----------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # -----------------------------------------------------
    # MATCH TRAINING COLUMNS
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")

    st.subheader("💰 Estimated Car Price")

    st.markdown(
        f"""
        <div style="
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            background-color: black;
            color: white;
        ">

        <h1>₹ {prediction:,.0f}</h1>

        <p>Estimated Price of the Car</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.subheader("📋 Dataset Information")

col1, col2 = st.columns(2)

with col1:

    st.write(
        "**Total Records:**",
        len(df)
    )

with col2:

    st.write(
        "**Features:**",
        len(X.columns)
    )


# =========================================================
# DATASET PREVIEW
# =========================================================

with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.caption(
    "Car Price Prediction | Random Forest Regression"
)
