# Emotion Classifier — Hybrid ML + Embedding + LLM System

This project uses the **dair-ai/emotion** dataset for educational purposes.  
Dataset: https://huggingface.co/datasets/dair-ai/emotion

### Citation  
Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).  
CARER: Contextualized Affect Representations for Emotion Recognition.  
EMNLP 2018.  
https://www.aclweb.org/anthology/D18-1404

---

This project implements a three-stage emotion analysis pipeline that combines:

1. A classical ML classifier for detecting emotions  
2. A sentence-embedding retrieval module for scientific grounding  
3. A controlled LLM (TinyLlama) for generating neutral, factual emotion explanations  

The goal is to take a user sentence, detect the underlying emotion, retrieve a relevant scientific context snippet, and produce a clear, consistent explanation.

---

## Project Structure

```
emotion-classifier-ml-project/
│
├── data/
│   ├── cleaned_dataset.csv
│   ├── label_mapping.json
│
├── knowledge/
│   ├── emotion.txt
│   ├── scientific_definitions.txt
│
├── logs/
│   ├── chatbot.log
│
├── src/
│   ├── chatbot_interface.py          # CLI interface
│   ├── config.py                     # central configuration
│   ├── emotion_classifier.py         # ML classifier + vectorizer + retrieval
│   ├── sentence_model.py             # helper for loading files
│   ├── ingest.py                     # dataset ingestion
│   ├── label_mapping.py              # loads label mapping JSON
│   ├── language_model.py             # TinyLlama interface
│   ├── simple_interface.py           # input cleaning utilities
│   ├── train_model.py                # training the ML classifier
│   │
│   ├── model.pkl                     # ML classification model (TF-IDF + classifier)
│   ├── sentense_model.pkl            # embedding cache for retrieval
│
└── test/
    └── pytest tests
├── .github/workflows/tests.yml           # CI workflow
├── requirements.txt
└── README.md
```
## Installation

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install all dependencies:

pip install --upgrade pip
pip install -r requirements.txt

## Pipeline Overview

The system operates in three distinct stages, each powered by one model.

---

### 1. Emotion Classification Model (model.pkl)

A trained ML model built using:

- TF-IDF vectorizer  
- A multiclass classifier (e.g., Logistic Regression)  

It predicts one of six categories:
sadness, joy, love, anger, fear, surprise


Implemented inside `emotion_classifier.py`.

---

### 2. Sentence Embedding Model (sentence_model.pkl)

Powered by:

sentence-transformers/all-MiniLM-L6-v2

Used to:

- Convert user text into embeddings  
- Retrieve relevant scientific definitions from the `knowledge/` directory  
- Provide factual grounding for the explanation  

This creates a minimal Retrieval-Augmented Generation (Mini-RAG) step.

---

### 3. TinyLlama LLM  
Model:

TinyLlama/TinyLlama-1.1B-Chat-v1.0


Purpose:

- Generate the final scientific explanation  
- Follow strict rules to avoid conversational tone  
- Avoid addressing the user directly  
- Avoid hallucinating stories or interpreting user meaning beyond classification  

The LLM receives:

- the detected emotion  
- one retrieved scientific chunk  
- strict formatting and style instructions  

Its output is a clean, emotion-specific explanation.

---

## Running the Emotion Classifier (CLI)

Start the application using:

python -m src.chatbot_interface


You will see:


Enter a sentence:


Example:


Enter a sentence: I feel nervous before my exam

Emotion detected: fear
Explanation:
Fear is a biological response associated with activation of the amygdala and stress-regulation circuits...

## Automated Testing
The project includes a complete pytest suite covering:
classifier loading and predictions
RAG semantic retrieval
chatbot generation
mapping correctness
ingest and cleaning functions
full pipeline integration test

Run all tests locally:

pytest -q

## Continuous Integration (GitHub Actions)
All tests run automatically on every push and pull request.
Workflow: .github/workflows/tests.yml