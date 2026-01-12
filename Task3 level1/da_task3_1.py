# ============================================
# DATA CLEANING PROJECT – TWO DATASETS
# ============================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 2. Load Datasets (FIXED PATH ISSUE)
# ============================================

df1 = pd.read_csv(
    r"C:\Users\Jayendra\Desktop\Task3 level1\AB_NYC_2019.csv"
)

df2 = pd.read_csv(
    r"C:\Users\Jayendra\Desktop\Task3 level1\FRvideos.csv"
)

print("Datasets Loaded Successfully!")

# ============================================
# 3. Basic Overview
# ============================================

print("\n--- Dataset 1 Info ---")
df1.info()

print("\n--- Dataset 2 Info ---")
df2.info()

# ============================================
# DATA CLEANING FUNCTION
# ============================================

def clean_dataset(df):
    print("\nInitial Shape:", df.shape)

    # 4. Handle Missing Values
    for col in df.columns:
        if df[col].dtype == "object":
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # 5. Remove Duplicate Records
    df = df.drop_duplicates()

    # 6. Standardization
    # Column names
    df.columns = df.columns.str.lower().str.strip()

    # String columns
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()

    # 7. Outlier Detection & Removal (IQR Method)
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print("Final Shape after Cleaning:", df.shape)
    return df

# ============================================
# APPLY CLEANING TO BOTH DATASETS
# ============================================

df1_cleaned = clean_dataset(df1)
df2_cleaned = clean_dataset(df2)

# ============================================
# DATA INTEGRITY CHECK
# ============================================

print("\nMissing Values in Dataset 1:")
print(df1_cleaned.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(df2_cleaned.isnull().sum())

# ============================================
# SAVE CLEANED DATASETS
# ============================================

df1_cleaned.to_csv(
    r"C:\Users\Jayendra\Desktop\Task3 level1\dataset1_cleaned.csv",
    index=False
)

df2_cleaned.to_csv(
    r"C:\Users\Jayendra\Desktop\Task3 level1\dataset2_cleaned.csv",
    index=False
)

print("\nCleaned datasets saved successfully!")
