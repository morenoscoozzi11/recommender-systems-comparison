"""
MODELLO 4: HYBRID Collaborative + Content-Based (senza LightFM)
Approccio: Combina predizioni SVD-like (matrix factorization via TruncatedSVD)
           con similarità coseno TF-IDF sulle review — approccio ibrido puro sklearn.

Sostituisce LightFM che non è compatibile con Python 3.12 (bug Cython ob_digit).

Pipeline:
  1. CF score:  TruncatedSVD sulla user-item matrix → predizione rating
  2. CBF score: TF-IDF + similarità coseno → predizione rating
  3. Score finale: media pesata CF + CBF (alpha configurabile)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

# ========== CONFIGURAZIONE ==========
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : os.path.join(DATA_DIR, 'amazon_with_text_features.csv'),
    'OUTPUT_PATH'         : os.path.join(OUT_DIR,  'results_hybrid_sklearn.csv'),
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,
    'THRESHOLD'           : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # SVD (CF component)
    'SVD_N_COMPONENTS'    : 20,
    # TF-IDF (CBF component)
    'MAX_FEATURES_TFIDF'  : 1000,
    # Peso ibrido: alpha * CF + (1-alpha) * CBF
    'ALPHA'               : 0.5,
}

# ========== METRICHE ==========

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Definizione standard:
      Precision@K = |{item in top-K con true_r >= threshold}| / K
      Recall@K    = |{item in top-K con true_r >= threshold}| / |{item totali con true_r >= threshold}|
    Il ranking è determinato dallo score stimato (est), non dal threshold.
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
    print("🔹 MODELLO 4: HYBRID sklearn (CF via TruncatedSVD + CBF via TF-IDF)")
    print("=" * 80)
    start_time = time.perf_counter()

    # ----- caricamento e pulizia -----
    print("\n📂 Caricamento dataset...")
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    print(f"   ✓ Dataset originale: {len(df)} righe, "
          f"{df['userID'].nunique()} utenti, {df['itemID'].nunique()} articoli")

    df = df.drop_duplicates(subset=['userID', 'itemID'], keep='first')

    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]

    df = df.groupby(['userID', 'itemID']).agg(
        rating=('rating', 'mean'),
        reviewText=('reviewText', 'first')
    ).reset_index()

    print(f"   ✓ Dopo filtro: {len(df)} righe, "
          f"{df['userID'].nunique()} utenti, {df['itemID'].nunique()} articoli")

    # ----- train / test split -----
    print(f"\n📊 Train/Test split 80/20...")
    train_df, test_df = train_test_split(
        df, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE']
    )
    print(f"   ✓ Train: {len(train_df)} | Test: {len(test_df)}")

    # ===================================================
    # COMPONENTE CF — TruncatedSVD sulla user-item matrix
    # ===================================================
    print(f"\n🤝 CF component: TruncatedSVD (n_components={CONFIG['SVD_N_COMPONENTS']})...")

    # matrice utente-item dal training
    interaction_matrix = train_df.pivot_table(
        index='userID', columns='itemID', values='rating', fill_value=0
    )
    user_list = list(interaction_matrix.index)
    item_list = list(interaction_matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}

    R = interaction_matrix.values   # shape: (n_users, n_items)

    n_components = min(CONFIG['SVD_N_COMPONENTS'], min(R.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=CONFIG['RANDOM_STATE'])
    U = svd.fit_transform(R)          # (n_users, k)
    Vt = svd.components_              # (k, n_items)
    R_approx = U @ Vt                 # ricostruzione (n_users, n_items)

    print(f"   ✓ SVD completato — explained variance: "
          f"{svd.explained_variance_ratio_.sum():.3f}")

    # ===================================================
    # COMPONENTE CBF — TF-IDF + cosine similarity
    # ===================================================
    print(f"\n📝 CBF component: TF-IDF (max_features={CONFIG['MAX_FEATURES_TFIDF']})...")

    item_text_df = (
        train_df[['itemID', 'reviewText']]
        .drop_duplicates('itemID')
        .set_index('itemID')
        .reindex(item_list)
        .fillna('')
    )
    vectorizer   = TfidfVectorizer(stop_words='english',
                                   max_features=CONFIG['MAX_FEATURES_TFIDF'])
    tfidf_matrix = vectorizer.fit_transform(item_text_df['reviewText'].values)
    sim_matrix   = cosine_similarity(tfidf_matrix)   # (n_items, n_items)
    print(f"   ✓ TF-IDF matrix: {tfidf_matrix.shape} | Sim matrix: {sim_matrix.shape}")

    # ===================================================
    # PREDIZIONI SUL TEST SET
    # ===================================================
    print(f"\n🎯 Generazione predizioni (alpha_CF={CONFIG['ALPHA']})...")

    global_mean = train_df['rating'].mean()

    # raggruppa test per utente: per le ranking metrics serve top-K per utente
    test_by_user = test_df.groupby('userID')

    all_predictions = []   # per ranking metrics (top-K per utente)
    rmse_true, rmse_pred = [], []  # per RMSE (predizione puntuale)

    for uid, user_test in test_by_user:
        # true ratings dell'utente nel test set
        true_ratings_map = dict(zip(user_test['itemID'], user_test['rating']))

        if uid not in user_to_idx:
            # utente cold-start: usa global_mean per RMSE
            for iid, true_r in true_ratings_map.items():
                all_predictions.append((uid, iid, true_r, global_mean, 0))
                rmse_true.append(true_r)
                rmse_pred.append(global_mean)
            continue

        u_idx        = user_to_idx[uid]
        user_ratings = interaction_matrix.loc[uid]
        rated_items  = user_ratings[user_ratings > 0]

        # --- calcola CBF scores una volta per tutti gli item ---
        cbf_scores = np.full(len(item_list), global_mean)
        w_sums = np.zeros(len(item_list))
        w_tots = np.zeros(len(item_list))
        for rated_item, r_val in rated_items.items():
            if rated_item in item_to_idx:
                j_idx = item_to_idx[rated_item]
                w_sums += sim_matrix[:, j_idx] * r_val
                w_tots += sim_matrix[:, j_idx]
        mask = w_tots > 0
        cbf_scores[mask] = np.clip(w_sums[mask] / w_tots[mask], 1.0, 5.0)

        # --- CF scores per tutti gli item ---
        cf_scores = np.clip(R_approx[u_idx, :], 1.0, 5.0)

        # --- score ibrido per tutti gli item ---
        hybrid_scores = CONFIG['ALPHA'] * cf_scores + (1 - CONFIG['ALPHA']) * cbf_scores

        # --- top-K per ranking metrics (usa true_rating del test se disponibile) ---
        top_k_idx = np.argsort(-hybrid_scores)[:CONFIG['K']]
        for rank, i_idx in enumerate(top_k_idx):
            pred_item = item_list[i_idx]
            # true rating: se l'item è nel test usa il vero valore, altrimenti 0
            true_r = true_ratings_map.get(pred_item, 0.0)
            all_predictions.append((uid, pred_item, true_r, hybrid_scores[i_idx], rank))

        # --- predizione puntuale per RMSE (solo item nel test) ---
        for iid, true_r in true_ratings_map.items():
            if iid in item_to_idx:
                i_idx = item_to_idx[iid]
                rmse_true.append(true_r)
                rmse_pred.append(hybrid_scores[i_idx])
            else:
                rmse_true.append(true_r)
                rmse_pred.append(global_mean)

    print(f"   ✓ {len(all_predictions)} predizioni generate")

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
        'Model'           : 'Hybrid-sklearn (CF+CBF)',
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
