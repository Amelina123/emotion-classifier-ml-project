import pandas as pd
from src.config import CLEAN_DATA_PATH, MODEL_PATH
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
import pickle


logging.basicConfig(level=logging.INFO)

    
df = pd.read_csv(CLEAN_DATA_PATH)
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f"Train size:, {X_train.shape}")
logging.info(f"Test size:, {X_test.shape}")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
logging.info(f"Vectorised train shape: {X_train_vec.shape}")
logging.info(f"Vectorised test shape: {X_test_vec.shape}")


model = LogisticRegression(max_iter=300,solver='lbfgs', class_weight="balanced", C=1.0)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
logging.info("\nAccuracy: %.4f", accuracy)
logging.info("\nClassification Report:\n%s", report)
logging.info("\nConfusion Matrix:\n%s", cm)



with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "vectorizer": vectorizer,
            "model": model},f)
logging.info(f"Model saved to {MODEL_PATH}")


df = pd.read_csv("data/cleaned_dataset.csv")
print(df["label"].unique())
print(df[["label", "text"]].drop_duplicates())
