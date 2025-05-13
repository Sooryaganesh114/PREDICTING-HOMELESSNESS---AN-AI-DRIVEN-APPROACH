import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix

# Step 1: Read the dataset from the CSV file
df = pd.read_csv('D:/miniproject/project_bayangaram/homelessness_risk_data.csv')

# Step 2: Visualize Feature Distributions
plt.figure(figsize=(12, 5))

# Income distribution by homelessness risk
plt.subplot(1, 2, 1)
sns.histplot(data=df, x="income", hue="risk_of_homelessness", kde=True, bins=30)
plt.title("Income Distribution by Homelessness Risk")

# Age distribution by homelessness risk
plt.subplot(1, 2, 2)
sns.histplot(data=df, x="age", hue="risk_of_homelessness", kde=True, bins=30)
plt.title("Age Distribution by Homelessness Risk")

plt.tight_layout()
plt.show()

# Step 3: Define Features and Target
features = df.drop(columns=['risk_of_homelessness'])
target = df['risk_of_homelessness']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Step 5: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability scores for the positive class

# Step 7: Evaluation Metrics

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Step 8: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Homelessness Risk Prediction")
plt.legend(loc="lower right")
plt.show()
