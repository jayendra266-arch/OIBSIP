import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

from wordcloud import WordCloud

# ===============================
# 1. LOAD DATA
# ===============================
apps = pd.read_csv("apps.csv")
reviews = pd.read_csv("user_reviews.csv")

print("Datasets Loaded Successfully")

# ===============================
# 2. DATA CLEANING
# ===============================
apps.drop_duplicates(inplace=True)

# Convert Rating to numeric
apps['Rating'] = pd.to_numeric(apps['Rating'], errors='coerce')

# Convert Reviews
apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')

# Clean Installs
apps['Installs'] = apps['Installs'].str.replace('[+,]', '', regex=True)
apps['Installs'] = pd.to_numeric(apps['Installs'], errors='coerce')

# Clean Price
apps['Price'] = apps['Price'].str.replace('$', '', regex=False)
apps['Price'] = pd.to_numeric(apps['Price'], errors='coerce')

# Clean Size
def convert_size(size):
    if isinstance(size, str):  # Only process strings
        size = size.strip()
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024
        elif size == 'Varies with device':
            return np.nan
    return np.nan  # Return NaN for floats or unhandled values

apps['Size'] = apps['Size'].apply(convert_size)

print("Data Cleaning Completed")

# ===============================
# 3. CATEGORY EXPLORATION
# ===============================
plt.figure(figsize=(10,6))
apps['Category'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.show()

# ===============================
# 4. METRICS ANALYSIS
# ===============================

# Ratings Distribution
plt.figure(figsize=(8,5))
sns.histplot(apps['Rating'], bins=20, kde=True, color='green')
plt.title("App Ratings Distribution")
plt.show()

# Free vs Paid Apps
plt.figure(figsize=(6,4))
apps['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'orange'])
plt.title("Free vs Paid Apps")
plt.ylabel("")
plt.show()

# Installs vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(x='Rating', y='Installs', data=apps, alpha=0.6)
plt.yscale('log')
plt.title("Rating vs Installs")
plt.show()

# Price Distribution
plt.figure(figsize=(8,5))
sns.boxplot(x=apps['Price'], color='lightcoral')
plt.title("Price Distribution of Apps")
plt.show()

# ===============================
# 5. SENTIMENT ANALYSIS
# ===============================
reviews = reviews.dropna(subset=['Translated_Review'])

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

reviews['Sentiment'] = reviews['Translated_Review'].apply(get_sentiment)

# Sentiment Distribution
plt.figure(figsize=(6,4))
reviews['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("User Review Sentiment Distribution")
plt.show()

# ===============================
# 6. WORD CLOUD
# ===============================
positive_reviews = reviews[reviews['Sentiment'] == 'Positive']['Translated_Review']
text = " ".join(positive_reviews)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Positive Review Word Cloud")
plt.show()

# ===============================
# 7. KEY INSIGHTS
# ===============================
print("\nKey Insights:")
print("1. Most apps are free, indicating ad-based monetization.")
print("2. Higher ratings generally correlate with higher installs.")
print("3. Paid apps are fewer but often better rated.")
print("4. User sentiment is predominantly positive.")
print("5. App sizes vary widely; many apps 'vary with device'.")
