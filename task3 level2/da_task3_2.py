# =====================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------
# 1. LOAD DATASET
# -------------------------------------
df = pd.read_csv("creditcard.csv")

print("\nDataset Loaded Successfully")
print(df.head())

# -------------------------------------
# 2. DATASET INFO
# -------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df["Class"].value_counts())

# -------------------------------------
# 3. VISUALIZE CLASS IMBALANCE
# -------------------------------------
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Legit Transactions")
plt.xlabel("Class (0 = Legit, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# -------------------------------------
# 4. FEATURE & TARGET SPLIT
# -------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------------
# 5. FEATURE SCALING (CORRECT WAY)
# -------------------------------------
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

X["Time"] = time_scaler.fit_transform(X[["Time"]])
X["Amount"] = amount_scaler.fit_transform(X[["Amount"]])

# -------------------------------------
# 6. TRAIN-TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------
# 7. MODEL TRAINING
# -------------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)
print("\nModel Training Completed")

# -------------------------------------
# 8. MODEL PREDICTION
# -------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------
# 9. MODEL EVALUATION
# -------------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------
# 10. REAL-TIME FRAUD PREDICTION (FIXED)
# -------------------------------------
def predict_fraud(transaction):
    transaction_df = pd.DataFrame([transaction], columns=X.columns)

    # Apply correct scalers
    transaction_df["Time"] = time_scaler.transform(transaction_df[["Time"]])
    transaction_df["Amount"] = amount_scaler.transform(transaction_df[["Amount"]])

    prediction = model.predict(transaction_df)

    return "🚨 Fraud Transaction" if prediction[0] == 1 else "✅ Legit Transaction"

# -------------------------------------
# 11. TEST SAMPLE TRANSACTION
# -------------------------------------
sample_transaction = df.drop("Class", axis=1).iloc[0].values

print("\nSample Transaction Prediction:")
print(predict_fraud(sample_transaction))
