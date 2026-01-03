import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ======================
# Configuration
# ======================
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "models/roberta_welfake"
MAX_LENGTH = 512

# ======================
# 1. Charger WELFake depuis Hugging Face et créer split test si nécessaire
# ======================
dataset = load_dataset("davanstrien/WELFake")
print("\n--- Splits initiaux ---")
print(dataset)

# Vérifier si test split existe, sinon créer 10% du train pour test
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
    print("\n--- Splits après création du test ---")
    print(dataset)

print(f"\nNombre d'exemples : Train = {len(dataset['train'])}, Test = {len(dataset['test'])}")

# ======================
# 2. Prétraitement du texte
# ======================
def clean_text(text):
    """Nettoyage basique du texte avec gestion des valeurs nulles"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(batch):
    cleaned_texts = [clean_text(t) for t in batch["text"]]
    tokens = tokenizer(
        cleaned_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = batch["label"]
    return tokens

# ======================
# 3. Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = dataset.map(preprocess, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ======================
# 4. Modèle Transformer
# ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ======================
# 5. Fonction metrics
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ======================
# 6. Entraînement
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# ======================
# 7. Évaluation
# ======================
print("\n==============================")
print(" ÉVALUATION DU MODÈLE ")
print("==============================")

predictions = trainer.predict(dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix :")
print(cm)

print("\nClassification Report :")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

# ======================
# 8. Sauvegarde du modèle
# ======================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning WELFake terminé avec succès.")
