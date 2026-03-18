"""
MODELLO 2: GENSIM Content-Based Filtering (Doc2Vec)
Approccio: Doc2Vec per rappresentazioni semantiche profonde delle review
Valutazione: Train/Test split 80/20, metriche standardizzate
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import gensim.utils
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️  Gensim non disponibile. Installa con: pip install gensim --break-system-packages")

# ========== CONFIGURAZIONE ==========
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : f'{DATA_DIR}/amazon_with_text_features.csv',
    'OUTPUT_PATH'         : f'{OUT_DIR}/results_gensim.csv',
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,
    'THRESHOLD'           : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # Doc2Vec
    'VECTOR_SIZE' : 100,
    'WINDOW'      : 5,
    'MIN_COUNT'   : 1,
    'EPOCHS'      : 40,
    'WORKERS'     : 4,
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
    if not GENSIM_AVAILABLE:
        print("❌ Gensim non disponibile. Installa con: pip install gensim --break-system-packages")
        return

    print("=" * 80)
    print("🔹 MODELLO 2: GENSIM (Content-Based Filtering — Doc2Vec)")
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

    # ----- Doc2Vec training -----
    print(f"\n📝 Training Doc2Vec "
          f"(vector_size={CONFIG['VECTOR_SIZE']}, epochs={CONFIG['EPOCHS']})...")

    # lista ordinata degli item nel training
    item_text_df = (
        train_df[['itemID', 'reviewText']]
        .drop_duplicates('itemID')
        .reset_index(drop=True)
    )
    item_list   = list(item_text_df['itemID'])
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}

    tagged_docs = [
        TaggedDocument(
            words=gensim.utils.simple_preprocess(row['reviewText']),
            tags=[str(row['itemID'])]
        )
        for _, row in item_text_df.iterrows()
    ]
    print(f"   ✓ {len(tagged_docs)} documenti preparati")

    model_d2v = Doc2Vec(
        vector_size=CONFIG['VECTOR_SIZE'],
        window=CONFIG['WINDOW'],
        min_count=CONFIG['MIN_COUNT'],
        workers=CONFIG['WORKERS'],
        epochs=CONFIG['EPOCHS'],
        seed=CONFIG['RANDOM_STATE']
    )
    model_d2v.build_vocab(tagged_docs)
    model_d2v.train(
        tagged_docs,
        total_examples=model_d2v.corpus_count,
        epochs=model_d2v.epochs
    )
    print(f"   ✓ Doc2Vec addestrato")

    # matrice embeddings: righe allineate con item_list
    item_vectors = np.array([model_d2v.dv[str(item)] for item in item_list])
    print(f"   ✓ Embeddings shape: {item_vectors.shape}")

    # matrice similarità coseno tra tutti gli item (n_items x n_items)
    sim_matrix = cosine_similarity(item_vectors)  # pre-calcolata una sola volta
    print(f"   ✓ Similarity matrix: {sim_matrix.shape}")

    # ----- matrice interazioni del training -----
    interaction_matrix = train_df.pivot_table(
        index='userID', columns='itemID', values='rating', fill_value=0
    )

    # ----- predizioni -----
    print(f"\n🎯 Generazione predizioni sul test set...")
    all_predictions     = []
    rmse_true, rmse_pred = [], []

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

        # --- score ranking: aggregazione ponderata su tutti gli item ---
        pred_scores = np.zeros(len(item_list))
        for rated_item, r_val in rated_items.items():
            if rated_item not in item_to_idx:
                continue
            r_idx = item_to_idx[rated_item]
            pred_scores += sim_matrix[r_idx] * r_val

        top_k_indices = np.argsort(-pred_scores)[:CONFIG['K']]
        for rank, idx in enumerate(top_k_indices):
            all_predictions.append(
                (uid, item_list[idx], true_rating, pred_scores[idx], rank)
            )

        # --- predizione rating per RMSE (media pesata dei top-K vicini del test_item) ---
        if test_item in item_to_idx:
            t_idx   = item_to_idx[test_item]
            sims    = sim_matrix[t_idx]              # similarità con tutti gli item
            # ordina per similarità decrescente, escludi se stesso
            sorted_idx = np.argsort(-sims)
            w_sum, w_tot = 0.0, 0.0
            for idx in sorted_idx[:CONFIG['K'] + 1]:
                nbr_item = item_list[idx]
                if nbr_item == test_item:
                    continue
                if nbr_item in rated_items.index:
                    sim = sims[idx]
                    w_sum += sim * rated_items[nbr_item]
                    w_tot += sim
            if w_tot > 0:
                rmse_true.append(true_rating)
                rmse_pred.append(w_sum / w_tot)

    print(f"   ✓ {len(all_predictions)} predizioni ranking generate")
    print(f"   ✓ {len(rmse_true)} predizioni rating generate (per RMSE)")

    if not all_predictions:
        print("   ❌ Nessuna predizione generata.")
        return

    # ----- metriche -----
    print(f"\n📈 Calcolo metriche...")
    precision, recall, f1 = precision_recall_at_k(
        all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD']
    )
    ndcg = ndcg_at_k(all_predictions, k=CONFIG['K'], threshold=CONFIG['THRESHOLD'])

    if rmse_true:
        rmse = float(np.sqrt(mean_squared_error(rmse_true, rmse_pred)))
        mae  = float(np.mean(np.abs(np.array(rmse_true) - np.array(rmse_pred))))
    else:
        rmse = mae = float('nan')

    execution_time = time.perf_counter() - start_time

    # ----- risultati -----
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI")
    print("=" * 80)

    results = {
        'Model'          : 'Gensim (CBF-Doc2Vec)',
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
