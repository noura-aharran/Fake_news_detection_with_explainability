# Fake_news_detection_with_explainability

Project 3 – Fake News Detection with Explainability
1. Type de données nécessaires
Pour entraîner et évaluer un modèle de détection de fake news :
1. Corpus de news/articles/post sociaux annotés
o Chaque texte est étiqueté comme "fake" ou "real".
o Exemples de datasets connus :
▪ LIAR dataset (courtes affirmations politiques avec labels vrai/faux).
https://aclanthology.org/P17-2067/
https://sites.cs.ucsb.edu/~william/papers/acl2017.pdf
https://datasets.activeloop.ai/docs/ml/datasets/liar-dataset
https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset
▪ FakeNewsNet (articles complets avec metadata).
https://arxiv.org/abs/1809.01286
https://github.com/KaiDMML/FakeNewsNet
https://asu.elsevierpure.com/en/datasets/fakenewsnet
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FUEMMHS
▪ Kaggle Fake News Challenge dataset (titre + contenu + étiquette).
https://www.kaggle.com/datasets/abhinavkrjha/fake-news-challenge
2. Sources multimodales (optionnel, avancé)
o Texte + titre + métadonnées (auteur, date, site web).
o Réactions sociales (likes, retweets, commentaires) → utiles pour détecter la
propagation d’une fake news.
3. Pré-traitements
o Nettoyage du texte (ponctuation, stopwords, lemmatisation).
o Tokenisation avec BERT / Transformers si tu vises le deep learning.
2. Méthodes utilisées
Le pipeline typique sera :
✓ Feature extraction
o TF-IDF, n-grams, embeddings (Word2Vec, GloVe, FastText).
o Pour plus avancé → BERT, RoBERTa, DistilBERT.
✓ Classification (détection)
o Modèles classiques : Logistic Regression, SVM, Random Forest.
o Modèles avancés : Transformers (fine-tuning de BERT ou RoBERTa).
✓ Explainability (explicabilité)
o Objectif : pas seulement dire "fake" ou "real", mais aussi pourquoi.
o Outils possibles :
▪ LIME / SHAP → montre quels mots ou phrases influencent la décision.
▪ Attention visualization → visualiser les poids d’attention dans BERT (les
mots les plus influents).
▪ Counterfactual explanations → proposer des alternatives ("si ce mot
n’était pas là, le modèle aurait prédit autre chose").
3. Résultat final attendu
À la fin du projet, tu devrais obtenir :
✓ Un modèle de détection
o Input : un article ou un post.
o Output : Probabilité que ce soit fake vs réel.
✓ Un module d’explicabilité
o Montrer quels éléments du texte ont conduit à la prédiction.
o Exemple :
▪ Texte : "Un médicament miracle guérit le cancer en 3 jours".
▪ Prédiction : 92% Fake.
▪ Explication : mots-clés suspects → ["miracle", "guérit", "3 jours"].
✓ Une évaluation complète
o Mesures : Accuracy, Precision, Recall, F1-score.
o Pour l’explicabilité : qualité perçue par les utilisateurs (utile/pertinent).
4. Applications concrètes
• Journalisme & médias → Outils de fact-checking automatique.
• Réseaux sociaux → Détection en temps réel de fake news virales.
• Recherche en IA → Combinaison NLP + XAI (très demandée aujourd’hui).
