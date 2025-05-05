import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load and split the dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Diabetes Score Predictor", layout="centered")
st.title("🧠 Diabetes Progression Predictor")
st.write("⚠️ These features are **normalized values**, not real-world units.")

# Input fields
user_input = []
for feature in data.feature_names:
    val = st.number_input(f"{feature} (normalized)", value=float(X[feature].mean()), format="%.4f")
    user_input.append(val)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=data.feature_names)
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Disease Progression Score: **{prediction:.2f}**")

# Show model quality
with st.expander("📈 Model Evaluation Metrics"):
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
