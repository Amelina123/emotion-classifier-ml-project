from src.label_mapping import load_label_mapping
import json
from src.config import LABEL_MAPPING_PATH


def test_mapping_is_correct_format():
    mapping = load_label_mapping()
    assert isinstance(mapping, dict)
    assert all(isinstance(k, int) for k in mapping.keys())
    assert all(isinstance(v, str) for v in mapping.values())
    assert len(mapping) >= 2  

