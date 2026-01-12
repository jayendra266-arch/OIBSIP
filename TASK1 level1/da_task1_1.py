import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# DATA LOADING
sales_df = pd.read_csv("retail_sales_dataset.csv")
menu_df = pd.read_csv("menu.csv")

print("Sales Dataset Preview:")
print(sales_df.head(), "\n")

print("Menu Dataset Preview:")
print(menu_df.head(), "\n")

#  DATA CLEANING
print("Missing values (Sales Dataset):")
print(sales_df.isnull().sum(), "\n")

print("Missing values (Menu Dataset):")
print(menu_df.isnull().sum(), "\n")

# Drop missing values
sales_df.dropna(inplace=True)
menu_df.dropna(inplace=True)

# Remove duplicates
sales_df.drop_duplicates(inplace=True)
menu_df.drop_duplicates(inplace=True)

# Convert Date column
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

print("Data Cleaning Completed.\n")


# DESCRIPTIVE STATISTICS

print("Descriptive Statistics (Sales Dataset):")
print(sales_df.describe(), "\n")

mean_sales = sales_df['Total Amount'].mean()
median_sales = sales_df['Total Amount'].median()
mode_sales = sales_df['Total Amount'].mode()[0]
std_sales = sales_df['Total Amount'].std()

print("Sales Statistics:")
print("Mean:", mean_sales)
print("Median:", median_sales)
print("Mode:", mode_sales)
print("Standard Deviation:", std_sales, "\n")


#  TIME SERIES ANALYSIS

# Monthly sales aggregation
monthly_sales = sales_df.groupby(sales_df['Date'].dt.to_period("M"))['Total Amount'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()


# CUSTOMER ANALYSIS


# Sales by Gender
gender_sales = sales_df.groupby('Gender')['Total Amount'].sum()

plt.figure()
gender_sales.plot(kind='bar')
plt.title("Sales by Gender")
plt.ylabel("Total Sales")
plt.show()

# Sales by Age Group
sales_df['Age Group'] = pd.cut(
    sales_df['Age'],
    bins=[0, 18, 30, 45, 60, 100],
    labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

age_group_sales = sales_df.groupby(
    'Age Group', observed=True
)['Total Amount'].sum()


plt.figure()
age_group_sales.plot(kind='bar')
plt.title("Sales by Age Group")
plt.ylabel("Total Sales")
plt.show()


#  PRODUCT ANALYSIS


product_sales = (
    sales_df.groupby('Product Category')['Total Amount']
    .sum()
    .sort_values(ascending=False)
)

plt.figure()
product_sales.plot(kind='bar')
plt.title("Sales by Product Category")
plt.ylabel("Total Sales")
plt.show()


# MENU DATA ANALYSIS (Dataset 2)


# Average calories per category
avg_calories = (
    menu_df.groupby('Category')['Calories']
    .mean()
    .sort_values(ascending=False)
)

plt.figure()
avg_calories.plot(kind='bar')
plt.title("Average Calories by Menu Category")
plt.ylabel("Calories")
plt.show()


# VISUALIZATION


# Sales Distribution
plt.figure()
sns.histplot(sales_df['Total Amount'], kde=True)
plt.title("Sales Amount Distribution")
plt.xlabel("Total Amount")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(
    sales_df[['Quantity', 'Price per Unit', 'Total Amount', 'Age']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()


#  BUSINESS RECOMMENDATIONS

print("========= BUSINESS RECOMMENDATIONS =========")
print("1. Electronics and Clothing categories generate the highest revenue.")
print("2. Adult and Middle-Age customers contribute most to total sales.")
print("3. Monthly sales trends help in demand-based inventory planning.")
print("4. High-calorie menu categories can be targeted for premium pricing.")
print("5. Quantity and Total Amount show strong positive correlation.")

print("\nEDA Completed Successfully.")
