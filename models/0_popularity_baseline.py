"""
MODELLO 0: POPULARITY BASELINE
================================
Il modello più semplice possibile: raccomanda a tutti gli utenti
gli item più popolari del training set.

Scopo nella tesi:
  Fornisce un punto di riferimento "stupido" per contestualizzare
  i risultati degli altri modelli. Se un modello sofisticato non
  batte questa baseline, c'è qualcosa che non va.

Due varianti testate:
  - Popularity by COUNT  : i K item con più interazioni nel training
  - Popularity by SCORE  : i K item con miglior rating medio ponderato
                           (bayesian average — penalizza item con poche recensioni)

Valutazione: Train/Test split 80/20, stesse metriche degli altri modelli.
             Item già visti dall'utente nel training vengono esclusi.

OUTPUT:
  results_popularity_baseline.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : os.path.join(DATA_DIR, 'amazon_with_text_features.csv'),
    'OUTPUT_PATH'         : os.path.join(OUT_DIR,  'results_popularity_baseline.csv'),
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,
    'THRESHOLD'           : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # Bayesian average: C = numero minimo di voti per essere considerato
    # (item con poche recensioni vengono "tirati" verso la media globale)
    'BAYES_C'             : 10,
}

# ─── METRICHE (identiche agli altri modelli) ──────────────────────────────────

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, __ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls, f1s = {}, {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k      = ratings[:k]
        n_hits     = sum(1 for _, t in top_k  if t >= threshold)
        n_relevant = sum(1 for _, t in ratings if t >= threshold)
        prec = n_hits / k           if k > 0          else 0.0
        rec  = n_hits / n_relevant  if n_relevant > 0  else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        precisions[uid] = prec
        recalls[uid]    = rec
        f1s[uid]        = f1
    return precisions, recalls, f1s

def ndcg_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, __ in predictions:
        user_est_true[uid].append((est, true_r))
    ndcgs = {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        dcg  = sum(1/np.log2(i+2) for i,(_, t) in enumerate(ratings[:k])
                   if t >= threshold)
        nrel = sum(1 for _, t in ratings if t >= threshold)
        idcg = sum(1/np.log2(i+2) for i in range(min(k, nrel)))
        ndcgs[uid] = dcg/idcg if idcg > 0 else 0.0
    return ndcgs

def evaluate(all_predictions, rmse_true, rmse_pred, k, threshold, label):
    """Calcola e stampa tutte le metriche per una variante."""
    prec, rec, f1 = precision_recall_at_k(all_predictions, k, threshold)
    ndcg          = ndcg_at_k(all_predictions, k, threshold)
    rmse = float(np.sqrt(mean_squared_error(rmse_true, rmse_pred)))
    mae  = float(np.mean(np.abs(np.array(rmse_true) - np.array(rmse_pred))))
    results = {
        'Model'           : label,
        'Precision@10'    : round(np.mean(list(prec.values())), 4),
        'Recall@10'       : round(np.mean(list(rec.values())),  4),
        'F1@10'           : round(np.mean(list(f1.values())),   4),
        'NDCG@10'         : round(np.mean(list(ndcg.values())), 4),
        'RMSE'            : round(rmse, 4),
        'MAE'             : round(mae,  4),
    }
    return results

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("🔹 MODELLO 0: POPULARITY BASELINE")
    print("   Variante A — Count-based  (top item per numero interazioni)")
    print("   Variante B — Score-based  (top item per rating medio ponderato)")
    print("=" * 80)
    start_time = time.perf_counter()

    # ── Caricamento e pulizia ────────────────────────────────────────────────
    print("\n📂 Caricamento dataset...")
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    print(f"   ✓ Dataset originale: {len(df):,} righe, "
          f"{df['userID'].nunique()} utenti, "
          f"{df['itemID'].nunique()} item")

    df = df.drop_duplicates(subset=['userID','itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]
    df = df.groupby(['userID','itemID']).agg(
        rating=('rating','mean'),
        reviewText=('reviewText','first')
    ).reset_index()
    print(f"   ✓ Dopo filtro: {len(df):,} righe, "
          f"{df['userID'].nunique()} utenti, "
          f"{df['itemID'].nunique()} item")

    # ── Train / Test split ───────────────────────────────────────────────────
    print(f"\n📊 Train/Test split 80/20...")
    train_df, test_df = train_test_split(
        df, test_size=CONFIG['TEST_SIZE'],
        random_state=CONFIG['RANDOM_STATE']
    )
    print(f"   ✓ Train: {len(train_df):,} | Test: {len(test_df):,}")

    global_mean = train_df['rating'].mean()
    print(f"   ✓ Rating medio globale (training): {global_mean:.4f}")

    # ── Calcolo popolarità degli item ────────────────────────────────────────
    print(f"\n📈 Calcolo popolarità item...")

    item_stats = train_df.groupby('itemID').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    ).reset_index()

    # Variante A: score = numero di interazioni (count)
    item_stats['score_count'] = item_stats['count']

    # Variante B: Bayesian Average
    # formula: (C * global_mean + count * mean_rating) / (C + count)
    # penalizza item con poche recensioni tirandoli verso la media globale
    C = CONFIG['BAYES_C']
    item_stats['score_bayes'] = (
        (C * global_mean + item_stats['count'] * item_stats['mean_rating'])
        / (C + item_stats['count'])
    )

    # Top-K globali per le due varianti
    top_k_count = (item_stats.sort_values('score_count', ascending=False)
                   .head(CONFIG['K'] * 10)['itemID'].tolist())
    top_k_bayes = (item_stats.sort_values('score_bayes', ascending=False)
                   .head(CONFIG['K'] * 10)['itemID'].tolist())

    print(f"   ✓ Top-3 per COUNT : {top_k_count[:3]}")
    print(f"   ✓ Top-3 per BAYES : {top_k_bayes[:3]}")

    item_to_score_count = dict(zip(item_stats['itemID'],
                                   item_stats['score_count']))
    item_to_score_bayes = dict(zip(item_stats['itemID'],
                                   item_stats['score_bayes']))
    item_to_mean        = dict(zip(item_stats['itemID'],
                                   item_stats['mean_rating']))

    # ── Predizioni ───────────────────────────────────────────────────────────
    print(f"\n🎯 Generazione predizioni...")

    preds_count, preds_bayes = [], []
    rmse_true_c, rmse_pred_c = [], []
    rmse_true_b, rmse_pred_b = [], []

    for uid, group in test_df.groupby('userID'):
        true_map = dict(zip(group['itemID'], group['rating']))

        # Item già visti nel training — da escludere
        seen = set(train_df[train_df['userID'] == uid]['itemID'].tolist())

        # ── Variante A: Count ───────────────────────────────────────────────
        top_unseen_count = [it for it in top_k_count if it not in seen]
        top_unseen_count = top_unseen_count[:CONFIG['K']]
        # Se non bastano, completa con item random non visti
        if len(top_unseen_count) < CONFIG['K']:
            extra = [it for it in item_stats['itemID']
                     if it not in seen and it not in top_unseen_count]
            top_unseen_count += extra[:CONFIG['K'] - len(top_unseen_count)]

        for rank, it in enumerate(top_unseen_count):
            score = item_to_score_count.get(it, 0)
            true_r = true_map.get(it, 0.0)
            preds_count.append((uid, it, true_r, score, rank))

        for _, row in group.iterrows():
            iid = row['itemID']
            p = item_to_mean.get(iid, global_mean)
            rmse_true_c.append(row['rating']); rmse_pred_c.append(p)

        # ── Variante B: Bayesian Average ────────────────────────────────────
        top_unseen_bayes = [it for it in top_k_bayes if it not in seen]
        top_unseen_bayes = top_unseen_bayes[:CONFIG['K']]
        if len(top_unseen_bayes) < CONFIG['K']:
            extra = [it for it in item_stats['itemID']
                     if it not in seen and it not in top_unseen_bayes]
            top_unseen_bayes += extra[:CONFIG['K'] - len(top_unseen_bayes)]

        for rank, it in enumerate(top_unseen_bayes):
            score = item_to_score_bayes.get(it, global_mean)
            true_r = true_map.get(it, 0.0)
            preds_bayes.append((uid, it, true_r, score, rank))

        for _, row in group.iterrows():
            iid = row['itemID']
            p = item_to_score_bayes.get(iid, global_mean)
            rmse_true_b.append(row['rating']); rmse_pred_b.append(p)

    print(f"   ✓ {len(preds_count)} predizioni generate per ogni variante")

    # ── Metriche ─────────────────────────────────────────────────────────────
    print(f"\n📈 Calcolo metriche...")
    execution_time = time.perf_counter() - start_time

    res_count = evaluate(preds_count, rmse_true_c, rmse_pred_c,
                         CONFIG['K'], CONFIG['THRESHOLD'],
                         'Popularity (Count)')
    res_bayes = evaluate(preds_bayes, rmse_true_b, rmse_pred_b,
                         CONFIG['K'], CONFIG['THRESHOLD'],
                         'Popularity (Bayesian Avg)')

    res_count['Execution_Time_s'] = round(execution_time, 2)
    res_bayes['Execution_Time_s'] = round(execution_time, 2)
    res_count['Num_Predictions']  = len(preds_count)
    res_bayes['Num_Predictions']  = len(preds_bayes)

    # ── Risultati ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI")
    print("=" * 80)

    results_df = pd.DataFrame([res_count, res_bayes])
    print("\n" + results_df.to_string(index=False))
    print("\n" + "=" * 80)

    # ── Interpretazione automatica ────────────────────────────────────────────
    print("\n💡 INTERPRETAZIONE")
    print("   Questi risultati sono il vostro 'floor' minimo.")
    print("   Qualsiasi modello che non supera questa baseline")
    print("   non sta imparando nulla di utile dal dataset.")
    print(f"\n   Baseline NDCG@10 (Bayesian): {res_bayes['NDCG@10']:.4f}")
    print(f"   Baseline RMSE   (Bayesian): {res_bayes['RMSE']:.4f}")

    results_df.to_csv(CONFIG['OUTPUT_PATH'], index=False)
    print(f"\n✅ Risultati salvati: {CONFIG['OUTPUT_PATH']}")
    return res_count, res_bayes


if __name__ == '__main__':
    main()
