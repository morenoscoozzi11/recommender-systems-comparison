"""
MODELLO 3: COLLABORATIVE FILTERING via Matrix Factorization (sklearn NMF/SVD)
Approccio: Fattorizzazione della user-item matrix con NMF (Non-negative Matrix
           Factorization) — equivalente concettuale di SVD usato in Surprise,
           implementato con sklearn puro (nessuna dipendenza esterna).

Sostituisce scikit-surprise che non è compatibile con NumPy 2.x.

Pipeline:
  1. Costruisce la user-item rating matrix dal training set
  2. Applica NMF per ottenere fattori latenti utente e item
  3. Ricostruisce la matrice → predice rating mancanti
  4. Valuta con le metriche standard
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

# ========== CONFIGURAZIONE ==========
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : os.path.join(DATA_DIR, 'amazon_cf.csv'),
    'OUTPUT_PATH'         : os.path.join(OUT_DIR,  'results_surprise.csv'),
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,
    'THRESHOLD'           : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # NMF parameters (equivalente a SVD factors in Surprise)
    'N_COMPONENTS' : 20,    # numero di fattori latenti
    'MAX_ITER'     : 300,   # iterazioni massime
    'ALPHA_W'      : 0.1,   # regolarizzazione
}

# ========== METRICHE ==========

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Precision@K = |{item in top-K con true_r >= threshold}| / K
    Recall@K    = |{item in top-K con true_r >= threshold}| / |{item totali con true_r >= threshold}|
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, __ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls, f1s = {}, {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k      = ratings[:k]
        n_hits     = sum(1 for _, true in top_k  if true >= threshold)
        n_relevant = sum(1 for _, true in ratings if true >= threshold)

        prec = n_hits / k           if k > 0          else 0.0
        rec  = n_hits / n_relevant  if n_relevant > 0  else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

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
        dcg   = sum((1.0 / np.log2(i + 2)) for i, (_, true) in enumerate(ratings[:k])
                    if true >= threshold)
        n_rel = sum(1 for _, true in ratings if true >= threshold)
        idcg  = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_rel)))
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0.0

    return ndcgs


# ========== MAIN ==========

def main():
    print("=" * 80)
    print("🔹 MODELLO 3: COLLABORATIVE FILTERING — NMF (sklearn, no Surprise)")
    print("=" * 80)
    start_time = time.perf_counter()

    # ----- caricamento e pulizia -----
    print("\n📂 Caricamento dataset...")
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    df.columns = [c.strip() for c in df.columns]
    print(f"   ✓ Dataset originale: {len(df)} righe, "
          f"{df['userID'].nunique()} utenti, {df['itemID'].nunique()} articoli")

    df = df.drop_duplicates(subset=['userID', 'itemID'], keep='first')

    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]

    df = df.groupby(['userID', 'itemID'])['rating'].mean().reset_index()
    print(f"   ✓ Dopo filtro: {len(df)} righe, "
          f"{df['userID'].nunique()} utenti, {df['itemID'].nunique()} articoli")

    # ----- train / test split -----
    print(f"\n📊 Train/Test split 80/20...")
    train_df, test_df = train_test_split(
        df, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE']
    )
    print(f"   ✓ Train: {len(train_df)} | Test: {len(test_df)}")

    # ----- costruzione user-item matrix -----
    print(f"\n🔧 Costruzione user-item matrix...")
    interaction_matrix = train_df.pivot_table(
        index='userID', columns='itemID', values='rating', fill_value=0
    )
    user_list   = list(interaction_matrix.index)
    item_list   = list(interaction_matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    R = interaction_matrix.values.astype(np.float64)   # (n_users, n_items)
    print(f"   ✓ Matrix shape: {R.shape} — sparsità: "
          f"{(R == 0).sum() / R.size * 100:.1f}%")

    # ----- NMF (Non-negative Matrix Factorization) -----
    n_comp = min(CONFIG['N_COMPONENTS'], min(R.shape) - 1)
    print(f"\n🤖 Training NMF (n_components={n_comp}, max_iter={CONFIG['MAX_ITER']})...")

    nmf = NMF(
        n_components=n_comp,
        max_iter=CONFIG['MAX_ITER'],
        random_state=CONFIG['RANDOM_STATE'],
        alpha_W=CONFIG['ALPHA_W'],
        init='nndsvda',       # inizializzazione robusta
    )
    W = nmf.fit_transform(R)   # (n_users, k)
    H = nmf.components_        # (k, n_items)
    R_approx = W @ H           # ricostruzione (n_users, n_items)

    print(f"   ✓ NMF completato — reconstruction error: {nmf.reconstruction_err_:.4f}")

    # ----- predizioni per ranking (top-K per utente) e RMSE -----
    print(f"\n🎯 Generazione predizioni sul test set...")

    global_mean = train_df['rating'].mean()
    all_predictions = []
    rmse_true, rmse_pred = [], []

    for uid, group in test_df.groupby('userID'):
        if uid not in user_to_idx:
            for _, row in group.iterrows():
                all_predictions.append((uid, row['itemID'], row['rating'], global_mean, 0))
                rmse_true.append(row['rating'])
                rmse_pred.append(global_mean)
            continue

        u_idx = user_to_idx[uid]

        # score per tutti gli item del training
        cf_scores = np.clip(R_approx[u_idx, :], 1.0, 5.0)

        # top-K per ranking metrics
        top_k_idx = np.argsort(-cf_scores)[:CONFIG['K']]
        true_map  = dict(zip(group['itemID'], group['rating']))
        for rank, i_idx in enumerate(top_k_idx):
            pred_item = item_list[i_idx]
            true_r    = true_map.get(pred_item, 0.0)
            all_predictions.append((uid, pred_item, true_r, cf_scores[i_idx], rank))

        # predizione puntuale per RMSE
        for _, row in group.iterrows():
            iid = row['itemID']
            if iid in item_to_idx:
                pred = cf_scores[item_to_idx[iid]]
            else:
                pred = global_mean
            rmse_true.append(row['rating'])
            rmse_pred.append(pred)

    print(f"   ✓ {len(all_predictions)} predizioni ranking generate")

    if not all_predictions:
        print("   ❌ Nessuna predizione generata.")
        return

    # ----- metriche -----
    print(f"\n📈 Calcolo metriche...")
    precision, recall, f1 = precision_recall_at_k(
        all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD']
    )
    ndcg = ndcg_at_k(all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD'])
    rmse = float(np.sqrt(mean_squared_error(rmse_true, rmse_pred)))
    mae  = float(np.mean(np.abs(np.array(rmse_true) - np.array(rmse_pred))))
    execution_time = time.perf_counter() - start_time

    # ----- risultati -----
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI")
    print("=" * 80)

    results = {
        'Model'           : 'CF-NMF (sklearn)',
        'Precision@10'    : round(np.mean(list(precision.values())), 4),
        'Recall@10'       : round(np.mean(list(recall.values())), 4),
        'F1@10'           : round(np.mean(list(f1.values())), 4),
        'NDCG@10'         : round(np.mean(list(ndcg.values())), 4),
        'RMSE'            : round(rmse, 4),
        'MAE'             : round(mae, 4),
        'Execution_Time_s': round(execution_time, 2),
        'Num_Predictions' : len(all_predictions),
    }

    results_df = pd.DataFrame([results])
    print("\n" + results_df.to_string(index=False))
    print("\n" + "=" * 80)

    results_df.to_csv(CONFIG['OUTPUT_PATH'], index=False)
    print(f"\n✅ Risultati salvati: {CONFIG['OUTPUT_PATH']}")
    return results


if __name__ == '__main__':
    main()
