# =====================================================
# CUSTOMER SEGMENTATION PROJECT
# Dataset: ifood_df.csv
# =====================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 2. Load Dataset
# Update the path according to your Windows system
file_path = r"C:\Users\Jayendra\Desktop\TASK2 level1\ifood_df.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file does not exist at path: {file_path}")

df = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(df.head())

# 3. Data Exploration
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Cleaning
# Fill missing numeric values with median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop duplicates if any
df = df.drop_duplicates()

# 5. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Example metrics
if 'purchase_amount' in df.columns:
    print("\nAverage Purchase Value:", round(df['purchase_amount'].mean(), 2))
    print("Total Purchases:", df['purchase_amount'].sum())
    print("Number of Transactions:", df.shape[0])

# 6. Feature Selection for Clustering
# Select only numeric features for clustering
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    raise ValueError("Need at least 2 numeric columns for clustering and visualization.")
X = df[numeric_cols]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Determine Optimal Clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 8. K-Means Clustering
optimal_clusters = 4  # Adjust based on elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['Customer_Segment'] = kmeans.fit_predict(X_scaled)

# 9. Segment Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=df[numeric_cols[0]],
    y=df[numeric_cols[1]],
    hue='Customer_Segment',
    data=df,
    palette='Set2',
    s=100
)
plt.title('Customer Segments')
plt.show()

# Optional: Pairplot for multi-dimensional visualization
sns.pairplot(df[numeric_cols + ['Customer_Segment']], hue='Customer_Segment', palette='Set2')
plt.show()

# 10. Insights and Recommendations
segment_summary = df.groupby('Customer_Segment')[numeric_cols].mean()
print("\nCustomer Segment Summary:")
print(segment_summary)

print("\nInsights Example:")
for i in range(optimal_clusters):
    print(f"Segment {i}:")
    print(segment_summary.loc[i])
    print("-"*40)
