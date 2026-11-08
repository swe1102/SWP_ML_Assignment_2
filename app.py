import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
  accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
  classification_report
)

st.set_page_config(page_title="Credit Card Default Model Evaluator", layout="wide")
st.title("Credit Card Default Prediction - Model Evaluator")

# 1. Upload Test CSV
st.subheader("1. Upload Test Data CSV")
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
  test_df = pd.read_csv(uploaded_file)
  st.write("Preview of Uploaded Data:", test_df.head(3))

  target_col = test_df.columns[-1]

  X_test = test_df.drop(columns=[target_col])
  y_test = test_df[target_col]

  # 2. Select Model
  st.subheader("2. Select Model")
  model_name = st.selectbox(
    "Choose a model to evaluate:",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
  )

  model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
  }

  if st.button("Evaluate Model"):
    model_path = model_files[model_name]
    try:
      model = joblib.load(model_path)
    except Exception as e:
      st.error(f"Could not load model file at {model_path}. Ensure it exists in your repository.")
      st.stop()

    y_pred = model.predict(X_test)

    try:
      y_prob = model.predict_proba(X_test)[:, 1]
      auc = roc_auc_score(y_test, y_prob)
    except Exception:
      auc = 0.0

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.success(f"Evaluation Complete for {model_name}!")

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    # Visualizations
    col_cm, col_cr = st.columns(2)
    with col_cm:
      st.subheader("Confusion Matrix")
      st.write(confusion_matrix(y_test, y_pred))

    with col_cr:
      st.subheader("Classification Report")
      st.text(classification_report(y_test, y_pred))
