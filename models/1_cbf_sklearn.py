"""
MODELLO 1: SCIKIT-LEARN Content-Based Filtering (TF-IDF + KNN)
Approccio: Estrae feature testuali tramite TF-IDF, calcola similarità coseno tra item
Valutazione: Train/Test split 80/20, metriche standardizzate
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

# ========== CONFIGURAZIONE ==========
# Modifica i path qui se necessario
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : f'{DATA_DIR}/amazon_with_text_features.csv',
    'OUTPUT_PATH'         : f'{OUT_DIR}/results_sklearn.csv',
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,       # top-k raccomandazioni
    'THRESHOLD'           : 4.0,      # soglia rating positivo
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    'MAX_FEATURES_TFIDF'  : 5000,
    'N_NEIGHBORS'         : 20,       # vicini KNN (>= K)
}

# ========== METRICHE ==========

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """Precision, Recall, F1 @k — predictions: list of (uid, iid, true_r, est, rank)"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, __ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls, f1s = {}, {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
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
    """NDCG @k"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, __ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        dcg  = sum((1.0 / np.log2(i + 2)) for i, (_, true) in enumerate(ratings[:k])
                   if true >= threshold)
        n_rel = sum(1 for _, true in ratings if true >= threshold)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_rel)))
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0.0

    return ndcgs


# ========== MAIN ==========

def main():
    print("=" * 80)
    print("🔹 MODELLO 1: SCIKIT-LEARN (Content-Based Filtering — TF-IDF + KNN)")
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

    # media rating per coppia user-item
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

    # ----- TF-IDF su tutti gli item del training -----
    print(f"\n📝 Estrazione TF-IDF (max_features={CONFIG['MAX_FEATURES_TFIDF']})...")

    # lista ordinata e stabile degli item nel training set
    item_list  = list(train_df['itemID'].unique())
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}

    # testo per ogni item (prima occorrenza)
    item_text = (
        train_df[['itemID', 'reviewText']]
        .drop_duplicates('itemID')
        .set_index('itemID')['reviewText']
        .to_dict()
    )
    texts = [item_text.get(item, '') for item in item_list]

    vectorizer  = TfidfVectorizer(
        stop_words='english',
        max_features=CONFIG['MAX_FEATURES_TFIDF']
    )
    tfidf_matrix = vectorizer.fit_transform(texts)   # shape: (n_items, n_features)
    print(f"   ✓ TF-IDF matrix: {tfidf_matrix.shape}")

    # ----- KNN -----
    n_neighbors = min(CONFIG['N_NEIGHBORS'], len(item_list))
    print(f"\n🔍 Training KNN (n_neighbors={n_neighbors}, metric=cosine)...")
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    knn.fit(tfidf_matrix)
    print(f"   ✓ KNN addestrato")

    # ----- matrice interazioni del training -----
    interaction_matrix = train_df.pivot_table(
        index='userID', columns='itemID', values='rating', fill_value=0
    )

    # ----- predizioni sul test set -----
    print(f"\n🎯 Generazione predizioni sul test set...")
    all_predictions     = []   # per ranking metrics
    rmse_true, rmse_pred = [], []  # per RMSE/MAE: rating medio dei vicini

    for _, row in test_df.iterrows():
        uid         = row['userID']
        test_item   = row['itemID']
        true_rating = row['rating']

        if uid not in interaction_matrix.index:
            continue

        user_ratings = interaction_matrix.loc[uid]
        rated_items  = user_ratings[user_ratings > 0]

        if rated_items.empty:
            continue

        # --- score per ranking (tutti gli item del training) ---
        pred_scores = np.zeros(len(item_list))
        for rated_item, r_val in rated_items.items():
            if rated_item not in item_to_idx:
                continue
            idx = item_to_idx[rated_item]
            distances, neighbors = knn.kneighbors(tfidf_matrix[idx])
            for dist, nbr_idx in zip(distances[0], neighbors[0]):
                pred_scores[nbr_idx] += (1.0 - dist) * r_val

        top_k_indices = np.argsort(-pred_scores)[:CONFIG['K']]
        for rank, idx in enumerate(top_k_indices):
            all_predictions.append(
                (uid, item_list[idx], true_rating, pred_scores[idx], rank)
            )

        # --- predizione rating per RMSE (media pesata dei vicini del test_item) ---
        if test_item in item_to_idx:
            t_idx = item_to_idx[test_item]
            distances, neighbors = knn.kneighbors(tfidf_matrix[t_idx])
            w_sum, w_tot = 0.0, 0.0
            for dist, nbr_idx in zip(distances[0], neighbors[0]):
                nbr_item = item_list[nbr_idx]
                if nbr_item == test_item:
                    continue
                if nbr_item in rated_items.index:
                    sim = 1.0 - dist
                    w_sum += sim * rated_items[nbr_item]
                    w_tot += sim
            if w_tot > 0:
                rmse_true.append(true_rating)
                rmse_pred.append(w_sum / w_tot)

    print(f"   ✓ {len(all_predictions)} predizioni ranking generate")
    print(f"   ✓ {len(rmse_true)} predizioni rating generate (per RMSE)")

    if not all_predictions:
        print("   ❌ Nessuna predizione generata. Controlla il dataset.")
        return

    # ----- metriche -----
    print(f"\n📈 Calcolo metriche...")
    precision, recall, f1 = precision_recall_at_k(
        all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD']
    )
    ndcg = ndcg_at_k(
        all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD']
    )

    if rmse_true:
        rmse = float(np.sqrt(mean_squared_error(rmse_true, rmse_pred)))
        mae  = float(np.mean(np.abs(np.array(rmse_true) - np.array(rmse_pred))))
    else:
        rmse = mae = float('nan')

    execution_time = time.perf_counter() - start_time

    # ----- stampa risultati -----
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI")
    print("=" * 80)

    results = {
        'Model'          : 'Scikit-Learn (CBF-TF-IDF)',
        'Precision@10'   : round(np.mean(list(precision.values())), 4),
        'Recall@10'      : round(np.mean(list(recall.values())), 4),
        'F1@10'          : round(np.mean(list(f1.values())), 4),
        'NDCG@10'        : round(np.mean(list(ndcg.values())), 4),
        'RMSE'           : round(rmse, 4),
        'MAE'            : round(mae, 4),
        'Execution_Time_s': round(execution_time, 2),
        'Num_Predictions': len(all_predictions),
    }

    results_df = pd.DataFrame([results])
    print("\n" + results_df.to_string(index=False))
    print("\n" + "=" * 80)

    results_df.to_csv(CONFIG['OUTPUT_PATH'], index=False)
    print(f"\n✅ Risultati salvati: {CONFIG['OUTPUT_PATH']}")
    return results


if __name__ == '__main__':
    main()
