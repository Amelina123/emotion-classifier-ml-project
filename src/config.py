from pathlib import Path
import os
from sentence_transformers import SentenceTransformer

HF_DATASET = "dair-ai/emotion"
DEFAULT_SPLIT = "train"
SPLITS = {
    "train": "split/train-00000-of-00001.parquet",
    "validation": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}

PATH_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PATH_ROOT/"data"/"cleaned_dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_MAPPING = "emotion"
LABEL_MAPPING_PATH = PATH_ROOT/"data"/'label_mapping.json'

MODEL_PATH = PATH_ROOT/"src"/"model.pkl"
LANGUAGE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge")

SENTENCE_MODEL_NAME ='all-MiniLM-L6-v2'
SENTENCE_MODEL_PATH = PATH_ROOT/"src"/"sentense_model.pkl"