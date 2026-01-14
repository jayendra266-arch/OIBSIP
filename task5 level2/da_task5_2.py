# 1. IMPORT LIBRARIES
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# 2. DATASET COLLECTION
corpus_text = """
machine learning is a field of artificial intelligence
machine learning models learn from data
natural language processing enables computers to understand text
deep learning is a subset of machine learning
artificial intelligence is transforming industries
"""

df = pd.DataFrame({'text': corpus_text.split('\n')})
df = df[df['text'].str.strip() != '']
print("Dataset Loaded Successfully")
print(df.head())

# 3. NLP PREPROCESSING (Regex Tokenizer)
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic
    tokens = text.split()  # simple split on spaces
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# Flatten tokens for unigram frequency
all_tokens = [token for sublist in df['tokens'] for token in sublist]

# 4. AUTOCOMPLETE (BIGRAM MODEL)
def build_bigram_model(tokens):
    bigram_model = defaultdict(list)
    for i in range(len(tokens)-1):
        bigram_model[tokens[i]].append(tokens[i+1])
    return bigram_model

bigram_model = build_bigram_model(all_tokens)

def autocomplete(prefix, model, top_k=5):
    predictions = Counter(model.get(prefix, []))
    return [word for word, _ in predictions.most_common(top_k)]

print("\nAutocomplete Test for 'machine':", autocomplete('machine', bigram_model))
print("Autocomplete Test for 'artificial':", autocomplete('artificial', bigram_model))

# 5. AUTOCORRECT (EDIT DISTANCE)
def edit_distance(a, b):
    # Simple iterative Levenshtein distance
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len(a)][len(b)]

vocab = set(all_tokens)

def autocorrect(word, vocabulary, max_distance=2):
    candidates = [(w, edit_distance(word, w)) for w in vocabulary if edit_distance(word, w) <= max_distance]
    candidates.sort(key=lambda x: x[1])
    if candidates:
        return candidates[0][0]
    else:
        return word

print("\nAutocorrect Test:")
print("lernng ->", autocorrect("lernng", vocab))
print("artifical ->", autocorrect("artifical", vocab))

# 6. METRICS
test_sentences = ["machine learning models", "artificial intelligence is transforming industries"]
total_next_words = 0
correct_predictions = 0
for sentence in test_sentences:
    words = preprocess_text(sentence)
    for i in range(len(words)-1):
        predicted_words = autocomplete(words[i], bigram_model, top_k=3)
        if words[i+1] in predicted_words:
            correct_predictions += 1
        total_next_words += 1
autocomplete_accuracy = correct_predictions / total_next_words
print("\nAutocomplete Accuracy (Top-3):", round(autocomplete_accuracy, 2))

test_pairs = [("lernng","learning"), ("artifical","artificial"), ("computrs","computers")]
correct = sum([autocorrect(wrong, vocab) == correct_word for wrong, correct_word in test_pairs])
autocorrect_accuracy = correct / len(test_pairs)
print("Autocorrect Accuracy:", round(autocorrect_accuracy, 2))

# 7. ALGORITHM COMPARISON
unigram_counts = Counter(all_tokens)
def autocomplete_unigram(prefix, top_k=5):
    return [word for word, _ in unigram_counts.most_common(top_k)]

print("\nAlgorithm Comparison:")
print("Unigram Autocomplete for 'machine':", autocomplete_unigram('machine'))
print("Bigram Autocomplete for 'machine':", autocomplete('machine', bigram_model))

# 8. VISUALIZATION
most_common_words = unigram_counts.most_common(10)
words, counts = zip(*most_common_words)

plt.figure(figsize=(8,5))
sns.barplot(x=list(words), y=list(counts))
plt.title("Top 10 Most Common Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

sample_word = 'machine'
next_words = Counter(bigram_model[sample_word])
plt.figure(figsize=(6,4))
sns.barplot(x=list(next_words.keys()), y=list(next_words.values()))
plt.title(f"Next-word Frequency after '{sample_word}'")
plt.xlabel("Next Word")
plt.ylabel("Count")
plt.show()
