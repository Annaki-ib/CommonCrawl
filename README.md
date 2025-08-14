# CommonCrawl
#  Contrastive Entity & Relation Analysis

##  Description
Ce projet implémente une **analyse contrastive** pour détecter et comparer les entités nommées ainsi que leurs relations dans un corpus textuel.  
Il exploite des données issues de **Common Crawl** et s’appuie sur des annotations pour entraîner et évaluer un modèle de classification **relation-rich** vs **relation-poor**.

##  Structure du dépôt

├── analysis_results_with_pos.jsonl # Résultats d'analyse enrichis avec POS tagging
├── COMMONcrawl.ipynb # Notebook d'analyse et d'expérimentation
├── contrastive_analysis_report.json # Rapport complet sur les statistiques
├── contrastive_learning_analysis.png # Visualisation du processus d'analyse
├── labeled_results.jsonl # Données annotées manuellement
├── sample_wet_subset.jsonl # Échantillon Common Crawl utilisé
├── README.md # Documentation du projet

## 📑 Contenu clé
- **contrastive_analysis_report.json**  
  Résumé statistique du corpus :
  - 17 documents analysés
  - 93 entités détectées (70 uniques)
  - Distribution `relation-rich` : 13 / `relation-poor` : 4
  - Entités les plus fréquentes : Fairbanks, Alaska, ParkWhiz, Knotts Berry Farm
  - Recommandations d’entraînement :  
    - `min_common_entities = 1`  
    - `batch_size = 8`  
    - `epochs = 5`  
    - `balance_ratio = 1.0`

- **analysis_results_with_pos.jsonl**  
  Résultats enrichis avec des **tags morphosyntaxiques (POS)** pour chaque entité.

- **labeled_results.jsonl**  
  Étiquetage manuel des relations entre entités.

- **sample_wet_subset.jsonl**  
  Sous-ensemble du corpus **Common Crawl WET**.

- **COMMONcrawl.ipynb**  
  Notebook Jupyter regroupant :
  - Prétraitement des données
  - Extraction des entités
  - Analyse des relations
  - Visualisations

- **contrastive_learning_analysis.png**  
  Illustration visuelle du pipeline d’analyse contrastive.

## ⚙️ Installation
```bash
# Cloner le repo
git clone https://github.com/username/nom-du-repo.git
cd nom-du-repo

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate
```

Utilisation

Explorer le notebook
Ouvrir COMMONcrawl.ipynb dans Jupyter pour reproduire l’analyse.

Analyser les rapports
Charger contrastive_analysis_report.json pour explorer les statistiques globales.

Entraîner un modèle contrastif
Utiliser les recommandations fournies dans le rapport pour définir vos hyperparamètres.

 Exemple de statistiques

Total documents : 17

Moyenne d'entités/document : 5.47

Classement entités fréquentes :

Fairbanks (5)

Alaska (5)

ParkWhiz (4)

Knotts Berry Farm (4)

Aperçu du pipeline
1. Installation et configuration des dépendances

Plusieurs cellules installent les bibliothèques nécessaires :

NLP & embeddings :

sentence-transformers, transformers → modèles de type BERT, RoBERTa, etc., pour embeddings et classification.

flair, spacy → NER, POS tagging.

fasttext, langdetect, pycld3 → détection de langue.

Parsing HTML & WARC/WET :

boilerpy3, justext, beautifulsoup4 → extraction du texte brut en supprimant le bruit HTML.

warcio, warc → lecture des fichiers WARC/WET de Common Crawl.

Utilitaires :

datasets → chargement d’échantillons Hugging Face.

boto3, botocore → accès à S3 (optionnel pour télécharger Common Crawl directement).

tqdm → barres de progression.

2. Ingestion & prétraitement des données

Source : fichiers WET (Web Extracted Text) de Common Crawl.

Alternative : chargement d’un petit sous-ensemble déjà prêt via datasets sur Hugging Face (évite de télécharger plusieurs Go).

Étapes de prétraitement :

Lecture ligne par ligne (JSONL).

Nettoyage du HTML avec boilerpy3 ou justext.

Détection de langue → garder uniquement une langue cible (ex. anglais).

Suppression des documents trop courts (MIN_WORDS) ou avec faible diversité lexicale (MIN_UNIQUE_RATIO).

3. Analyse linguistique

NER (Named Entity Recognition) :
Utilisation de spaCy ou flair pour extraire PERSON, ORG, LOC, etc.
Exemple : "Barack Obama visited Paris" → PERSON=Barack Obama, LOC=Paris.

POS tagging (Part-of-Speech) :
Détection des verbes, noms, adjectifs, etc. pour analyser les structures grammaticales.
Sert notamment à trouver des co-occurrences entités + actions.

4. Labellisation hybride (Weak Supervision)

Heuristiques :

Score basé sur densité d’entités nommées.

Pondération si entités et verbes apparaissent ensemble dans certaines structures.

Zero-shot classification :

Utilisation de modèles comme facebook/bart-large-mnli pour classer les documents dans des catégories sans entraînement spécifique.

Exemple : prompt → "Ce document parle-t-il de politique, sport, technologie ?"

5. Apprentissage auto-supervisé (Contrastive Learning)

Génération de paires :

Positives : deux documents avec un certain nombre d’entités en commun.

Négatives : documents sans entités partagées.

Entraînement :

Utilisation de modèles type sentence-transformers pour apprendre un espace vectoriel où les documents similaires (positifs) sont proches, et les différents (négatifs) sont éloignés.

6. Sorties attendues

Fichiers JSONL à chaque étape :

Données brutes nettoyées.

Résultats d’analyse (entités, POS tags).

Résultats labellisés (weak supervision).

Embeddings finaux après apprentissage contrastif.

