from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_PATH = "models/roberta_welfake_liar"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

explainer = LimeTextExplainer(class_names=["Fake", "Real"])

def predict_proba(texts):
    outputs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        outputs.append(probs)
    return np.array(outputs)

def explain(text):
    return explainer.explain_instance(
        text,
        predict_proba,
        num_features=10
    )
