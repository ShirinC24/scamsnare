import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# ------------------------------------
# Load Dataset
# ------------------------------------
# Using SMS Spam Collection Dataset
df = pd.read_csv("spam.csv", encoding='utf-8', on_bad_lines='skip')  # Skip problematic lines

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ------------------------------------
# Preprocessing function
# ------------------------------------
stop_words = stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["cleaned"] = df["message"].apply(preprocess_text)

# ------------------------------------
# Split data
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"], test_size=0.2, random_state=42
)

# ------------------------------------
# Vectorizer
# ------------------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------------
# Train Model
# ------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------------------------
# Test Accuracy
# ------------------------------------
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)

print(f"✅ Training Completed Successfully!")
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# ------------------------------------
# Save Model + Vectorizer
# ------------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ model.pkl and vectorizer.pkl saved successfully!")
