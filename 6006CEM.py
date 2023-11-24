# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('creditcard.csv')

# Extract features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Isolation Forest Implementation
# ===============================

# Create and fit the Isolation Forest model
isolation_forest_model = IsolationForest(contamination=0.005, random_state=42)
isolation_forest_model.fit(X_train)

# Predict on the test set
y_pred_isolation_forest = isolation_forest_model.predict(X_test)

# Convert predictions to binary (1 for inliers, -1 for outliers/frauds)
y_pred_isolation_forest_binary = np.where(y_pred_isolation_forest == 1, 0, 1)

# Evaluate the Isolation Forest model
print("Isolation Forest Results:")
print(classification_report(y_test, y_pred_isolation_forest_binary))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_isolation_forest_binary))

# ========================================
# Gradient Boosting (XGBoost) Implementation
# ========================================

# Create and fit the XGBoost model
xgboost_model = XGBClassifier(learning_rate=0.05, n_estimators=100, random_state=42)
xgboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgboost = xgboost_model.predict(X_test)

# Evaluate the XGBoost model
print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgboost))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgboost))

# =====================================
# Area Under the Precision-Recall Curve
# =====================================

# Calculate the average precision score for both models
average_precision_isolation_forest = average_precision_score(y_test, y_pred_isolation_forest_binary)
average_precision_xgboost = average_precision_score(y_test, y_pred_xgboost)

# Plot the precision-recall curves
precision_isolation_forest, recall_isolation_forest, _ = precision_recall_curve(y_test, y_pred_isolation_forest_binary)
precision_xgboost, recall_xgboost, _ = precision_recall_curve(y_test, y_pred_xgboost)

plt.figure(figsize=(10, 6))
plt.step(recall_isolation_forest, precision_isolation_forest, color='b', alpha=0.8, where='post', label='Isolation Forest (AP={:.2f})'.format(average_precision_isolation_forest))
plt.step(recall_xgboost, precision_xgboost, color='r', alpha=0.8, where='post', label='XGBoost (AP={:.2f})'.format(average_precision_xgboost))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

