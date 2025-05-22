# Credit Card Fraud Detection

This project aims to build a machine learning model to detect fraudulent credit card transactions using a highly imbalanced dataset.

## Objective

The primary goal is to develop a classification model capable of accurately identifying fraudulent transactions, while addressing challenges associated with class imbalance. The project focuses on data preprocessing, exploratory data analysis (EDA), model training, and evaluation using appropriate performance metrics.

## Dataset

- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: ~150 MB
- Features: 30 variables including anonymized features (V1 to V28), `Time`, `Amount`, and the target variable `Class` (0 = legitimate, 1 = fraud)

## Project Workflow

### 1. Data Preprocessing
- Standardized the `Amount` and `Time` features using `StandardScaler`
- Handled class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Split data into training and testing sets (80/20)

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution to highlight imbalance
- Created boxplots for feature distributions by class
- Generated correlation heatmap to assess feature relationships

### 3. Model Training
- Trained two classification models:
  - Logistic Regression
  - Random Forest (with hyperparameter tuning using RandomizedSearchCV)
- Evaluated models using test set

### 4. Model Evaluation
- Performance metrics used:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Visualized confusion matrix and ROC curves

## Key Findings

- SMOTE effectively balanced the training data and improved recall for the minority (fraud) class.
- Random Forest outperformed Logistic Regression in most metrics, especially F1-score and ROC-AUC.
- Accuracy was not used as a primary metric due to the highly imbalanced nature of the data.

## Future Improvements

- Experiment with other classifiers such as XGBoost or LightGBM
- Explore ensemble methods or stacking
- Apply anomaly detection techniques or autoencoders for unsupervised detection

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, imbalanced-learn

## How to Run

1. Install dependencies using pip:
pip install -r requirements.txt
2. Run the Jupyter Notebook or Python script:
jupyter notebook credit_card_fraud_detection.ipynb