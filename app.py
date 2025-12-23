import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DETAILS.csv")
    df = df.dropna()
    y = df["selling_price"]
    X = df.drop(["selling_price"], axis=1)
    X_encoded = pd.get_dummies(X, drop_first=True)
    return df, X, y, X_encoded
df, X, y, X_encoded = load_data()
@st.cache_resource
def train_model(X_encoded, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42
    )
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, r2
model, r2_score_test = train_model(X_encoded, y)
feature_columns = X_encoded.columns
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to estimate its selling price.")
st.sidebar.header("Car Input Details")
car_name = st.sidebar.selectbox("Car Name", df["name"].unique())
year = st.sidebar.slider(
    "Year of Manufacture",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=int(df["year"].median())
)
km_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=int(df["km_driven"].max()),
    value=int(df["km_driven"].median())
)
fuel = st.sidebar.selectbox("Fuel Type", df["fuel"].unique())
seller_type = st.sidebar.selectbox("Seller Type", df["seller_type"].unique())
transmission = st.sidebar.selectbox("Transmission", df["transmission"].unique())
owner = st.sidebar.selectbox("Owner Type", df["owner"].unique())
input_dict = {
    "name": [car_name],
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
}
input_df = pd.DataFrame(input_dict)
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
if st.button("Predict Car Price"):
    prediction = model.predict(input_encoded)[0]

    st.subheader("📌 Predicted Selling Price:")
    st.success(f"₹ {int(prediction):,}")
st.markdown("---")
#st.write("Model R² Score (test data):", round(r2_score_test, 3))
