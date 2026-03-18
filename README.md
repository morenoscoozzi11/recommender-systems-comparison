# Analisi Comparativa di Modelli di Raccomandazione

**Studente:** Moreno Scozzi  
**Tirocinio Universitario** | Università degli Studi di Udine  
**Dataset:** Amazon Movies & TV   
**Linguaggio:** Python 3.12

---

## Descrizione

Questo repository contiene il codice e i risultati sperimentali di un'analisi comparativa di 5 modelli di raccomandazione implementati sul dataset pubblico Amazon Movies & TV.

I modelli coprono le principali famiglie di approcci:

| # | Script | Approccio | Libreria |
|---|--------|-----------|----------|
| 1 | `models/1_cbf_sklearn.py` | Content-Based Filtering (TF-IDF + KNN) | scikit-learn |
| 2 | `models/2_cbf_gensim.py` | Content-Based Filtering (Doc2Vec) | gensim |
| 3 | `models/3_cf_nmf.py` | Collaborative Filtering (NMF) | scikit-learn |
| 4 | `models/4_hybrid_sklearn.py` | Hybrid CF + CBF (SVD + TF-IDF) | scikit-learn |
| 5 | `models/5_hybrid_tensorflow.py` | Hybrid Deep Learning (Embeddings + TF-IDF) | TensorFlow/Keras |

---

## Struttura del Repository

```
├── models/                  # Script dei 5 modelli
├── scripts/
│   ├── preprocess_amazon.py # Preprocessing del dataset grezzo (.txt.gz)
│   ├── 0_run_all_models.py  # Script master: esegue tutti i modelli
│   └── 6_compare_multidim.py# Confronto su 3 dimensioni del dataset
└── results/                 # CSV con i risultati sperimentali
```

---

## Dataset

Il dataset utilizzato è  **Amazon Product Reviews — Movies and TV**, disponibile su [UCSD Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

Il file di partenza è `movies.txt.gz`. Non è incluso nel repository per le sue dimensioni, va scaricato separatamente e posizionato nella stessa cartella degli script.




> **Nota:** TensorFlow richiede Python 3.9–3.12. Versione testata: `tensorflow==2.20.0`.  
> scikit-surprise **non è incluso** per incompatibilità con NumPy 2.x — sostituito con NMF di scikit-learn.

---

## Utilizzo

### 1. Preprocessing

Genera i CSV per le tre dimensioni del dataset (10k, 40k, 100k righe):

```bash
# Posiziona movies.txt.gz nella stessa cartella
python3 scripts/preprocess_amazon.py
```

Output: `amazon_cf_10k.csv`, `amazon_with_text_10k.csv`, (+ 40k, 100k)

### 2. Esecuzione singolo modello

```bash
python3 models/1_cbf_sklearn.py
```

Ogni script è autonomo e produce un CSV di risultati nella stessa cartella.

### 3. Esecuzione di tutti i modelli (script master)

```bash
python3 scripts/0_run_all_models.py
```

Output: `comparison_all_models.csv`

### 4. Confronto multi-dimensionale

```bash
python3 scripts/6_compare_multidim.py
```

Output: `comparison_multidim.csv` con i risultati di tutti i modelli su 10k, 40k e 100k righe.

---

## Metriche di Valutazione

Tutte le metriche sono calcolate con K=10 e soglia di rilevanza rating ≥ 4.0:

- **Precision@10** — proporzione di item rilevanti nei top-10 suggeriti
- **Recall@10** — proporzione di item rilevanti recuperati tra tutti quelli rilevanti
- **F1@10** — media armonica di Precision e Recall
- **NDCG@10** — Normalized Discounted Cumulative Gain (tiene conto dell'ordinamento)
- **RMSE** — Root Mean Squared Error sulla predizione numerica del rating
- **MAE** — Mean Absolute Error

---

## Risultati Principali (dataset 10k)

| Modello | P@10 | R@10 | NDCG@10 | RMSE | Tempo |
|---------|------|------|---------|------|-------|
| 1. Sklearn TF-IDF | 0.0093 | 0.0930 | 0.0645 | 1.181 | 0.29s |
| 2. Gensim Doc2Vec | 0.0116 | 0.0930 | 0.0353 | 1.122 | 1.42s |
| 3. CF NMF | 0.0163 | 0.1628 | 0.0575 | 3.085 | 0.07s |
| 4. Hybrid CF+CBF | 0.0023 | 0.0233 | 0.0116 | 1.834 | 0.05s |
| **5. TensorFlow DL** | **0.1333** | **0.7692** | **0.7598** | **0.957** | 2.80s |

Il modello TensorFlow DL risulta il più performante su tutte le metriche principali e migliora consistentemente con dataset più grandi (NDCG@10 = 0.85 a 100k righe).

---

## Problemi di Compatibilità Risolti

| Libreria originale | Problema | Soluzione |
|--------------------|----------|-----------|
| scikit-surprise | Incompatibile con NumPy 2.x | Sostituita con NMF di scikit-learn |
| LightFM | Non compila su Python 3.12 (Cython) | Reimplementato con SVD + TF-IDF sklearn |
| tensorflow-recommenders | Richiede Keras 2 (ambiente: Keras 3) | Reimplementato con TF/Keras puro |
| tensorflow 2.21 | Conflitto con tf-keras | Fissato a tensorflow==2.20.0 |
