import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

MODEL_PATH = "models/roberta_welfake_liar"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits, dim=1)[0].tolist()
    label = int(torch.argmax(outputs.logits))

    return {
        "label": label,
        "probabilities": probs
    }
