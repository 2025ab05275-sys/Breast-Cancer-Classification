# Breast-Cancer-Classification

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a tumor is benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset. 

The goal is to evaluate different models using multiple performance metrics and determine the best-performing model.


## b. Dataset Description

The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

- Number of instances: 569
- Number of features: 30 numerical features
- Target variable:
  - B = Benign
  - M = Malignant

The dataset contains computed features from digitized images of breast mass cell nuclei.


## c. Models Used

The following models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Comparison Table of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble)| 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble)| 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## Observations on Model Performance

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Performed very well with high AUC and balanced precision-recall performance. |
| Decision Tree | Lower performance compared to other models, possibly due to overfitting. |
| KNN | Good performance but slightly lower recall compared to ensemble methods. |
| Naive Bayes | Moderate performance due to feature independence assumption. |
| Random Forest (Ensemble) | Best overall performance with highest MCC and perfect precision. |
| XGBoost (Ensemble) | Excellent performance with highest AUC and strong generalization. |



