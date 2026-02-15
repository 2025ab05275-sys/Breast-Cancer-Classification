import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "KNN",
         "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if model_choice == "Logistic Regression":
        model = joblib.load("model/logistic_regression.pkl")
    elif model_choice == "Decision Tree":
        model = joblib.load("model/decision_tree.pkl")
    elif model_choice == "KNN":
        model = joblib.load("model/knn.pkl")
    elif model_choice == "Naive Bayes":
        model = joblib.load("model/naive_bayes.pkl")
    elif model_choice == "Random Forest":
        model = joblib.load("model/random_forest.pkl")
    else:
        model = joblib.load("model/xgboost.pkl")

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.write("### Accuracy:", round(acc, 4))

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
