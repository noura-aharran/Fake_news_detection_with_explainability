import shap
from transformers import pipeline

MODEL_PATH = "models/roberta_welfake_liar"

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    return_all_scores=True
)

explainer = shap.Explainer(classifier)

def explain(text):
    shap_values = explainer([text])
    return shap_values
