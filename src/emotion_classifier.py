import logging
import pickle
from src.config import MODEL_PATH, SENTENCE_MODEL_PATH
from src.label_mapping import load_label_mapping
from src.external_files import embed_knowledge

class EmotionClassifier:
    def __init__(self, model_path=MODEL_PATH):
        logging.info(f"Loading model from: {MODEL_PATH}")

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.vectorizer = saved["vectorizer"]
        self.model = saved["model"]
        logging.info("Model and vectorizer loaded successfully.")

        self.mapping = load_label_mapping()
        logging.info(f"Loaded label mapping: {self.mapping}")

        with open(SENTENCE_MODEL_PATH, "rb") as f:
            knowledge_data = pickle.load(f)
        self.knowledge_embeddings = knowledge_data["embeddings"]
        self.knowledge_chunks = knowledge_data["chunks"]
        logging.info("Knowledge embeddings and chunks loaded.")
    
    
    
    
    def classify(self, text: str):
        vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)
        label_id = int(prediction[0])
        emotion_name = self.mapping[label_id]
        return emotion_name
    

