# customer-churn-prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load Dataset
file_path = "bigml_59c28831336c6604c800002a.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})
df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].astype(int)
df = df.drop(columns=['state', 'phone number', 'area code'])

# Define features and target
X = df.drop(columns=['churn'])
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline Model: Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# Evaluate Logistic Regression
log_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_log),
    "Precision": precision_score(y_test, y_pred_log),
    "Recall": recall_score(y_test, y_pred_log),
    "F1-Score": f1_score(y_test, y_pred_log),
    "ROC AUC": roc_auc_score(y_test, y_prob_log)
}

# Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# Evaluate Decision Tree
tree_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_tree),
    "Precision": precision_score(y_test, y_pred_tree),
    "Recall": recall_score(y_test, y_pred_tree),
    "F1-Score": f1_score(y_test, y_pred_tree),
    "ROC AUC": roc_auc_score(y_test, y_prob_tree)
}

# Hyperparameter Tuning for Decision Tree
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model from Grid Search
best_tree_model = grid_search.best_estimator_
y_pred_best_tree = best_tree_model.predict(X_test)
y_prob_best_tree = best_tree_model.predict_proba(X_test)[:, 1]

# Evaluate Tuned Decision Tree
tuned_tree_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_best_tree),
    "Precision": precision_score(y_test, y_pred_best_tree),
    "Recall": recall_score(y_test, y_pred_best_tree),
    "F1-Score": f1_score(y_test, y_pred_best_tree),
    "ROC AUC": roc_auc_score(y_test, y_prob_best_tree)
}

# Summary of Results
print("Logistic Regression:", log_metrics)
print("Decision Tree:", tree_metrics)
print("Tuned Decision Tree:", tuned_tree_metrics)

# Conclusion & Next Steps
# - Decision Tree performed better than Logistic Regression in identifying churn
# - Hyperparameter tuning improved the balance between precision and recall
# - Next steps: Try ensemble models (Random Forest, Gradient Boosting) and handle class imbalance (SMOTE)
