# train_transformer_fake_news.py
import os
import random
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

# -------- CONFIG --------
MODEL_NAME = "roberta-base"        # tu peux tester "distilbert-base-uncased", "deberta-v3-small", etc.
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128
OUTPUT_DIR = "./saved_model"
SEED = 42

# fichiers CSV/JSON (remplace par tes chemins)
FILES = {
    "liar": "data/liar.csv",           # contient au minimum 'text' et 'label' (peut être 6-classes)
    "fakenewsnet": "data/fakenewsnet.csv", 
    "gossipcop": "data/gossipcop.csv",
    "welfake": "data/welfake.csv",
    "snopes": "data/snopes.csv",
    # ajoute d'autres fichiers si besoin
}

# Mapping labels hétérogènes -> labels standard (0/1/2) ou garder multi-classes pour LIAR
LABEL_MAP_SIMPLE = {
    # exemples
    "pants-fire": 0, "false": 0, "fake": 0,
    "barely-true": 1, "half-true": 1, "mixture": 1, "unverified": 1,
    "mostly-true": 2, "true": 2, "real": 2
}
# Si LIAR à 6 classes et tu veux garder 6 classes, tu remplaceras LABEL_MAP_SIMPLE par un mapping différent.

# -------- utilities --------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_table(path):
    # charge CSV ou JSON (détecte automatiquement)
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=str).fillna("")
    elif path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        raise ValueError("Format non supporté pour " + path)

def unify_df(df, text_cols=None, label_col_candidates=None, source_name=None):
    # essaye d'extraire une colonne texte et une colonne label
    if not text_cols:
        # colonnes possibles
        for c in ["text", "claim", "statement", "content", "article", "body"]:
            if c in df.columns:
                text_cols = [c]; break
    if not text_cols:
        raise ValueError("Impossible de trouver une colonne texte dans le df")
    text_col = text_cols[0]
    label_col = None
    if label_col_candidates:
        for c in label_col_candidates:
            if c in df.columns:
                label_col = c; break
    else:
        for c in ["label", "truthLabel", "veracity", "fact_check", "stance"]:
            if c in df.columns:
                label_col = c; break
    if label_col is None:
        # si pas de labels, on met 'unknown'
        df["_raw_label"] = "unknown"
    else:
        df["_raw_label"] = df[label_col].astype(str)
    df2 = pd.DataFrame({
        "text": df[text_col].astype(str),
        "raw_label": df["_raw_label"].astype(str),
        "source": source_name or "",
    })
    return df2

# -------- main pipeline --------
def main():
    seed_everything()

    dfs = []
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] fichier absent: {path} — je saute {name}")
            continue
        df = load_table(path)
        # tu peux adapter text_cols/label_col_candidates si la structure est différente
        df_un = unify_df(df, text_cols=None, label_col_candidates=None, source_name=name)
        dfs.append(df_un)
        print(f"Chargé {name}: {len(df_un)} lignes")

    if len(dfs) == 0:
        raise SystemExit("Aucun dataset chargé. Place tes CSV/JSON dans data/ et réessaie.")

    # concat
    big_df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print("Total après concat:", len(big_df))

    # map labels (simple) — si 'unknown' -> on peut drop ou marquer -1
    def map_label_raw(x):
        k = x.strip().lower()
        return LABEL_MAP_SIMPLE.get(k, "unknown")
    big_df["label_mapped"] = big_df["raw_label"].apply(map_label_raw)
    print("Labels uniques après mapping:", big_df["label_mapped"].unique())

    # filtering: optionnel -> garder uniquement label != 'unknown'
    big_df = big_df[big_df["label_mapped"] != "unknown"].reset_index(drop=True)
    # cast int
    big_df["label"] = big_df["label_mapped"].astype(int)

    # split train / val / test
    train_df, test_df = train_test_split(big_df, test_size=0.10, random_state=SEED, stratify=big_df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1111, random_state=SEED, stratify=train_df["label"])  # ~80/10/10
    print("Splits sizes:", len(train_df), len(val_df), len(test_df))

    # convertir en HuggingFace datasets
    ds_train = Dataset.from_pandas(train_df[["text","label","source"]])
    ds_val = Dataset.from_pandas(val_df[["text","label","source"]])
    ds_test = Dataset.from_pandas(test_df[["text","label","source"]])
    dataset_dict = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    dataset_tokenized = dataset_dict.map(tokenize_fn, batched=True, remove_columns=["text","source"])

    # model
    num_labels = len(set(dataset_tokenized["train"]["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        seed=SEED,
        fp16=torch.cuda.is_available()
    )

    # metrics
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
        return {"accuracy": acc, "f1": f1}

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # entraînement
    trainer.train()

    # évaluation finale
    print("Evaluation test set")
    res = trainer.evaluate(dataset_tokenized["test"])
    print(res)

    # sauvegarde
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
