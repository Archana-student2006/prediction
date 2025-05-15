import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Boston House Price Predictor")

# Load and prepare the data
@st.cache_data
def load_data():
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    df = boston.frame
    df = df.apply(pd.to_numeric)
    return df

df = load_data()
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"*Model Performance:*")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Input sliders
st.sidebar.header("Input Features")

def user_input():
    inputs = {
        col: st.sidebar.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        for col in X.columns
    }
    return pd.DataFrame([inputs])

input_df = user_input()

# Make prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction * 1000:.2f}")

# Visualization: Actual vs Predicted
if st.checkbox("Show Prediction Chart"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, color='purple', ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)
