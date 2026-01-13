import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset (CSV file in same folder)
df = pd.read_csv("Housing.csv")

print("Dataset Loaded Successfully\n")
print(df.head())


# Dataset structure
print("\nDataset Information:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)



# Convert categorical variables into numerical form
df_encoded = pd.get_dummies(df, drop_first=True)

# Target variable: price
y = df_encoded["price"]

# Feature variables: all other columns
X = df_encoded.drop("price", axis=1)

print("\nSelected Features:")
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully")


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
