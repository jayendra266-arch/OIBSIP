import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

#  Load Dataset
df = pd.read_csv('WineQT.csv')  
print("Dataset Loaded Successfully!")
print(df.head())

# Drop the Id column
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

#  Basic Dataset Info
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())

#  Visualizations
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x='quality', data=df)
plt.title("Distribution of Wine Quality")
plt.show()

#  Features and Target
X = df.drop('quality', axis=1)
y = df['quality']  # Multi-class target

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Model Training
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# SGD Classifier
sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train_scaled, y_train)
sgd_pred = sgd_model.predict(X_test_scaled)

# Support Vector Classifier
svc_model = SVC(kernel='rbf', random_state=42)
svc_model.fit(X_train_scaled, y_train)
svc_pred = svc_model.predict(X_test_scaled)

#  Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    print(f"===== {model_name} =====")
    print("Accuracy Score:", accuracy_score(y_true, y_pred))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

#  Evaluate Models
evaluate_model(y_test, rf_pred, "Random Forest Classifier")
evaluate_model(y_test, sgd_pred, "SGD Classifier")
evaluate_model(y_test, svc_pred, "Support Vector Classifier")
