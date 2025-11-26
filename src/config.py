from pathlib import Path


DEFAULT_SPLIT = "train"
HF_DATASET = "dair-ai/emotion"
SPLITS = {
    "train": "split/train-00000-of-00001.parquet",
    "validation": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}

PATH_ROOT = Path(__file__).resolve().parent.parent
FILENAME_PATH = PATH_ROOT / "data" / "cleaned_dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"