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
    confusion_matrix
)

st.title("Breast Cancer Classification App")

st.markdown("Evaluate using test data!")

# Download Sample Test Data Section
st.subheader("Download Sample Test Dataset")

try:
    sample_data = pd.read_csv("model/test_data.csv")
    st.download_button(
        label="Download Sample Test CSV",
        data=sample_data.to_csv(index=False),
        file_name="test_data.csv",
        mime="text/csv"
    )
except:
    st.info("Sample test dataset not available.")

st.divider()

# Upload Section
st.subheader("Upload Test CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Model Selection
st.subheader("Select Model")
model_name = st.selectbox(
    "",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Run Button
run_button = st.button("Run Evaluation")


# Prediction Logic
if run_button:

    if uploaded_file is None:
        st.warning("Please upload a CSV file before running evaluation.")
    else:
        data = pd.read_csv(uploaded_file)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        model_files = {
            "Logistic Regression": "model/logistic_regression.pkl",
            "Decision Tree": "model/decision_tree.pkl",
            "KNN": "model/knn.pkl",
            "Naive Bayes": "model/naive_bayes.pkl",
            "Random Forest": "model/random_forest.pkl",
            "XGBoost": "model/xgboost.pkl"
        }

        model = joblib.load(model_files[model_name])

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # ======================
        # Metrics Table
        # ======================
        st.subheader("Evaluation Metrics")

        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "AUC Score": roc_auc_score(y, y_prob),
            "MCC Score": matthews_corrcoef(y, y_pred)
        }

        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        st.table(metrics_df)

    
        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
