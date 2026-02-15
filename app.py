import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Breast Cancer Classification App")

# Dataset Upload
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Assuming last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Model Selection Dropdown
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

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    st.write("## Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # Confusion Matrix
    st.write("## Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.write("## Classification Report")
    st.text(classification_report(y, y_pred))
