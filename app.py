import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Breast Cancer Classification App")

st.write("Upload a CSV file containing test data (features + target column).")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # Load selected model
    if model_name == "Logistic Regression":
        model = joblib.load("model/logistic_regression.pkl")
    elif model_name == "Decision Tree":
        model = joblib.load("model/decision_tree.pkl")
    elif model_name == "KNN":
        model = joblib.load("model/knn.pkl")
    elif model_name == "Naive Bayes":
        model = joblib.load("model/naive_bayes.pkl")
    elif model_name == "Random Forest":
        model = joblib.load("model/random_forest.pkl")
    else:
        model = joblib.load("model/xgboost.pkl")

    # Predict
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

