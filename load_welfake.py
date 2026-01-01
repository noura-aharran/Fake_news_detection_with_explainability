import os
from datasets import load_dataset
from collections import Counter

# Supprime le warning Windows symlink
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Charger le dataset public WELFake
dataset = load_dataset("davanstrien/WELFake")

# Afficher les splits
print("\n--- Splits du dataset ---")
print(dataset.keys())

# Afficher les colonnes/features
print("\n--- Colonnes/features ---")
print(dataset['train'].column_names)

# Aperçu des 5 premiers exemples
print("\n--- 5 premiers exemples ---")
for ex in dataset['train'][:5]:
    print(ex)

# Compter le nombre d'articles fake vs real
labels = [ex['label'] for ex in dataset['train']]
label_count = Counter(labels)
print("\n--- Nombre d'articles par label ---")
print(f"Fake (0): {label_count.get(0,0)}")
print(f"Real (1): {label_count.get(1,0)}")
