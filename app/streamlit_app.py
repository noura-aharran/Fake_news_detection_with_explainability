import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# ======================
# Configuration page
# ======================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection with Explainability")
st.markdown("Analyse automatique des articles via **Transformers fine-tunés**")

# ======================
# Charger le modèle
# ======================
@st.cache_resource
def load_model():
    model_name = "models/roberta_welfake_liar"  # chemin vers ton modèle final
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ======================
# Zone de texte utilisateur
# ======================
text = st.text_area(
    "📝 Entrez un article ou une phrase :",
    height=250,
    placeholder="Copiez ici le contenu de l'article..."
)

# ======================
# Bouton prédiction
# ======================
if st.button("🔍 Analyser"):
    if text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probs = softmax(outputs.logits, dim=1).numpy()[0]
            prediction = np.argmax(probs)

            label_map = {0: "FAKE ❌", 1: "REAL ✅"}

        # ======================
        # Résultats
        # ======================
        st.subheader("📊 Résultat")
        st.markdown(f"### 🔎 Prédiction : **{label_map[prediction]}**")

        st.progress(float(probs[prediction]))

        st.write("**Probabilités :**")
        st.write(f"- Fake : `{probs[0]:.2f}`")
        st.write(f"- Real : `{probs[1]:.2f}`")

        # ======================
        # Explainability (placeholder)
        # ======================
        st.subheader("🧠 Explainability")
        st.info(
            "Les explications SHAP / LIME seront affichées ici "
            "dans la prochaine étape du projet."
        )

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("Projet Fake News Detection • Transformers • Streamlit")
