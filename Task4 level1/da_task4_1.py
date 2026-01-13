import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df1 = pd.read_csv("user_reviews.csv")
df2 = pd.read_csv("apps.csv")

print("Dataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)


# Pick first text-like column automatically
df1 = df1.iloc[:, 0].to_frame(name="text")
df2 = df2.iloc[:, 0].to_frame(name="text")

df = pd.concat([df1, df2], ignore_index=True)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("Combined Dataset Shape:", df.shape)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['clean_text'].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())


X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

nb_pred = nb.predict(X_test_tfidf)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

svm_pred = svm.predict(X_test_tfidf)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


def predict_sentiment(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    return svm.predict(vector)[0]

print("\nCustom Tests:")
print("I love this app →", predict_sentiment("I love this app"))
print("Worst app ever →", predict_sentiment("Worst app ever"))
print("It is okay →", predict_sentiment("It is okay"))
