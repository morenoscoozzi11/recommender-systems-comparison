"""
SCRIPT 9: RIPETIBILITÀ — Modelli 1-4 con 3 seed diversi
=========================================================
Obiettivo: dimostrare che i risultati sono stabili e non dipendono
           dalla casualità del train/test split.

Strategia:
  - 3 seed: 42, 123, 7 (stessi dell'ablation study per coerenza)
  - Dataset: amazon_with_text_40k.csv
  - Modelli: 1 (TF-IDF+KNN), 2 (Doc2Vec), 3 (CF-NMF), 4 (Hybrid SVD)
  - Modello 5 (TensorFlow): già coperto dall'ablation study (seed 42,123,7)

OUTPUT:
  reproducibility_results.csv   — tutti i risultati grezzi
  reproducibility_summary.csv   — media ± std per ogni modello
"""

import os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

DATASET_PATH = '/home/claude/amazon_with_text_40k.csv'
OUT_DIR      = '/mnt/user-data/outputs'
SEEDS        = [42, 123, 7]

CONFIG = {
    'TEST_SIZE'            : 0.2,
    'K'                    : 10,
    'THRESHOLD'            : 4.0,
    'MIN_USER_INT'         : 5,
    'MIN_ITEM_INT'         : 5,
    'TFIDF_MAX_FEATURES'   : 5000,
    'KNN_N_NEIGHBORS'      : 20,
    'D2V_VECTOR_SIZE'      : 100,
    'D2V_EPOCHS'           : 40,
    'NMF_N_COMPONENTS'     : 20,
    'NMF_MAX_ITER'         : 300,
    'SVD_N_COMPONENTS'     : 20,
    'HYBRID_ALPHA'         : 0.5,
    'HYBRID_TFIDF_FEATURES': 1000,
}

# ─── METRICHE ─────────────────────────────────────────────────────────────────

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
        precisions[uid] = prec; recalls[uid] = rec; f1s[uid] = f1
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

def compute_metrics(all_preds, rt, rp):
    prec, rec, f1 = precision_recall_at_k(
        all_preds, CONFIG['K'], CONFIG['THRESHOLD'])
    ndcg = ndcg_at_k(all_preds, CONFIG['K'], CONFIG['THRESHOLD'])
    rmse = float(np.sqrt(mean_squared_error(rt, rp)))
    mae  = float(np.mean(np.abs(np.array(rt) - np.array(rp))))
    return {
        'Precision@10': round(np.mean(list(prec.values())), 4),
        'Recall@10'   : round(np.mean(list(rec.values())),  4),
        'F1@10'       : round(np.mean(list(f1.values())),   4),
        'NDCG@10'     : round(np.mean(list(ndcg.values())), 4),
        'RMSE'        : round(rmse, 4),
        'MAE'         : round(mae,  4),
    }

# ─── CARICAMENTO ──────────────────────────────────────────────────────────────

def load_dataset(seed):
    df = pd.read_csv(DATASET_PATH)
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    df = df.drop_duplicates(subset=['userID','itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INT']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INT']].index)]
    df = df.groupby(['userID','itemID']).agg(
        rating=('rating','mean'),
        reviewText=('reviewText','first')
    ).reset_index()
    train, test = train_test_split(
        df, test_size=CONFIG['TEST_SIZE'], random_state=seed)
    return df, train, test

# ─── MODELLO 1: TF-IDF + KNN ──────────────────────────────────────────────────

def run_sklearn(train, test):
    t0 = time.perf_counter()
    item_list   = list(train['itemID'].unique())
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    item_text   = train.groupby('itemID')['reviewText'].first().to_dict()
    texts       = [item_text.get(it,'') for it in item_list]

    vec   = TfidfVectorizer(stop_words='english',
                            max_features=CONFIG['TFIDF_MAX_FEATURES'])
    tfidf = vec.fit_transform(texts)
    n_nb  = min(CONFIG['KNN_N_NEIGHBORS'], len(item_list)-1)
    knn   = NearestNeighbors(metric='cosine', algorithm='brute',
                             n_neighbors=n_nb)
    knn.fit(tfidf)

    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        rated = train[train['userID']==uid][['itemID','rating']]
        if rated.empty:
            for _, row in group.iterrows():
                all_preds.append((uid,row['itemID'],row['rating'],global_mean,0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        scores = np.zeros(len(item_list))
        weights = np.zeros(len(item_list))
        for _, r in rated.iterrows():
            if r['itemID'] not in item_to_idx: continue
            idx = item_to_idx[r['itemID']]
            dist, nbrs = knn.kneighbors(tfidf[idx], n_neighbors=n_nb)
            for d, ni in zip(dist[0], nbrs[0]):
                scores[ni]  += (1-d) * r['rating']
                weights[ni] += (1-d)
        mask = weights > 0
        pred = np.clip(np.where(mask, scores/weights, global_mean), 1, 5)
        true_map  = dict(zip(group['itemID'], group['rating']))
        top_k_idx = np.argsort(-pred)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it,0.0), pred[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = pred[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp)
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    return m

# ─── MODELLO 2: DOC2VEC ───────────────────────────────────────────────────────

def run_gensim(train, test):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import gensim
    t0 = time.perf_counter()

    item_text_df = (train[['itemID','reviewText']].drop_duplicates('itemID')
                    .set_index('itemID'))
    item_list   = list(item_text_df.index)
    item_to_idx = {it: i for i, it in enumerate(item_list)}

    tagged = [TaggedDocument(
                words=gensim.utils.simple_preprocess(row['reviewText']),
                tags=[str(iid)])
              for iid, row in item_text_df.iterrows()]

    model = Doc2Vec(vector_size=CONFIG['D2V_VECTOR_SIZE'], window=5,
                    min_count=1, epochs=CONFIG['D2V_EPOCHS'],
                    workers=1, seed=42)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

    vecs       = np.array([model.dv[str(it)] for it in item_list])
    sim_matrix = cosine_similarity(vecs)
    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        rated    = train[train['userID']==uid][['itemID','rating']]
        true_map = dict(zip(group['itemID'], group['rating']))
        if rated.empty:
            for _, row in group.iterrows():
                all_preds.append((uid,row['itemID'],row['rating'],global_mean,0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        w_sum = np.zeros(len(item_list))
        w_tot = np.zeros(len(item_list))
        for _, r in rated.iterrows():
            if r['itemID'] not in item_to_idx: continue
            j = item_to_idx[r['itemID']]
            w_sum += sim_matrix[:, j] * r['rating']
            w_tot += sim_matrix[:, j]
        pred = np.clip(np.where(w_tot > 0, w_sum/w_tot, global_mean), 1, 5)
        top_k_idx = np.argsort(-pred)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it,0.0), pred[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = pred[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp)
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    return m

# ─── MODELLO 3: CF NMF ────────────────────────────────────────────────────────

def run_nmf(train, test):
    t0 = time.perf_counter()
    matrix = train.pivot_table(index='userID', columns='itemID',
                                values='rating', fill_value=0)
    user_list   = list(matrix.index)
    item_list   = list(matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    R = matrix.values.astype(np.float64)

    n_comp = min(CONFIG['NMF_N_COMPONENTS'], min(R.shape)-1)
    nmf = NMF(n_components=n_comp, max_iter=CONFIG['NMF_MAX_ITER'],
              random_state=42, init='nndsvda')
    W = nmf.fit_transform(R)
    H = nmf.components_
    R_approx = np.clip(W @ H, 1.0, 5.0)

    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        true_map = dict(zip(group['itemID'], group['rating']))
        if uid not in user_to_idx:
            for _, row in group.iterrows():
                all_preds.append((uid,row['itemID'],row['rating'],global_mean,0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        u      = user_to_idx[uid]
        scores = R_approx[u, :]
        top_k_idx = np.argsort(-scores)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it,0.0), scores[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = scores[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp)
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    return m

# ─── MODELLO 4: HYBRID SVD+TF-IDF ────────────────────────────────────────────

def run_hybrid(train, test):
    t0 = time.perf_counter()
    matrix = train.pivot_table(index='userID', columns='itemID',
                                values='rating', fill_value=0)
    user_list   = list(matrix.index)
    item_list   = list(matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    R = matrix.values.astype(np.float64)

    n_comp = min(CONFIG['SVD_N_COMPONENTS'], min(R.shape)-1)
    svd  = TruncatedSVD(n_components=n_comp, random_state=42)
    U    = svd.fit_transform(R)
    Vt   = svd.components_
    R_cf = np.clip(U @ Vt, 1.0, 5.0)

    item_text_df = (train[['itemID','reviewText']].drop_duplicates('itemID')
                    .set_index('itemID').reindex(item_list).fillna(''))
    vec   = TfidfVectorizer(stop_words='english',
                            max_features=CONFIG['HYBRID_TFIDF_FEATURES'])
    tfidf = vec.fit_transform(item_text_df['reviewText'].values)
    sim   = cosine_similarity(tfidf)

    global_mean = train['rating'].mean()
    alpha = CONFIG['HYBRID_ALPHA']
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        true_map = dict(zip(group['itemID'], group['rating']))
        if uid not in user_to_idx:
            for _, row in group.iterrows():
                all_preds.append((uid,row['itemID'],row['rating'],global_mean,0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        u           = user_to_idx[uid]
        cf_scores   = R_cf[u, :]
        rated_row   = matrix.loc[uid]
        rated_items = rated_row[rated_row > 0]
        w_sum = np.zeros(len(item_list))
        w_tot = np.zeros(len(item_list))
        for ri, rv in rated_items.items():
            if ri not in item_to_idx: continue
            j = item_to_idx[ri]
            w_sum += sim[:, j] * rv
            w_tot += sim[:, j]
        cbf = np.where(w_tot > 0, np.clip(w_sum/w_tot,1,5), global_mean)
        hybrid    = alpha * cf_scores + (1-alpha) * cbf
        top_k_idx = np.argsort(-hybrid)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it,0.0), hybrid[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = hybrid[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp)
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    return m

# ─── MAIN ─────────────────────────────────────────────────────────────────────

MODELS = [
    ('1 - Sklearn TF-IDF', run_sklearn),
    ('2 - Gensim Doc2Vec', run_gensim),
    ('3 - CF NMF',         run_nmf),
    ('4 - Hybrid CF+CBF',  run_hybrid),
]

def main():
    print("=" * 70)
    print("🔬 RIPETIBILITÀ — Modelli 1-4 con 3 seed diversi")
    print(f"   Seeds: {SEEDS} | Dataset: 40k")
    print("=" * 70)

    all_results = []
    total = len(MODELS) * len(SEEDS)
    done  = 0

    for model_name, model_fn in MODELS:
        print(f"\n{'='*70}")
        print(f"📊 {model_name}")
        print(f"{'='*70}")

        for seed in SEEDS:
            done += 1
            print(f"   Seed {seed} ({done}/{total})...", end='', flush=True)
            try:
                _, train, test = load_dataset(seed)
                result = model_fn(train, test)
                result['Model'] = model_name
                result['Seed']  = seed
                all_results.append(result)
                print(f" ✓ NDCG={result['NDCG@10']:.4f} | "
                      f"RMSE={result['RMSE']:.4f} | "
                      f"t={result['Execution_Time_s']}s")
            except Exception as e:
                print(f" ❌ {e}")

    if not all_results:
        print("❌ Nessun risultato."); return

    # ─── SALVATAGGIO GREZZI ───────────────────────────────────────────────────
    cols_raw = ['Model','Seed','Precision@10','Recall@10','F1@10',
                'NDCG@10','RMSE','MAE','Execution_Time_s']
    df_raw = pd.DataFrame(all_results)[cols_raw]
    path_raw = os.path.join(OUT_DIR, 'reproducibility_results.csv')
    df_raw.to_csv(path_raw, index=False)

    # ─── SUMMARY: MEDIA ± STD ─────────────────────────────────────────────────
    metrics = ['Precision@10','Recall@10','F1@10','NDCG@10','RMSE','MAE']
    summary_rows = []
    for model_name, grp in df_raw.groupby('Model'):
        row = {'Model': model_name, 'n_runs': len(grp)}
        for m in metrics:
            row[f'{m}_mean'] = round(grp[m].mean(), 4)
            row[f'{m}_std']  = round(grp[m].std(),  4)
        summary_rows.append(row)

    # Aggiunge TF (da ablation study, seed 42/123/7 sul 40k)
    tf_data = {
        'Model'           : '5 - TensorFlow DL',
        'n_runs'          : 3,
        'Precision@10_mean': 0.1577, 'Precision@10_std': 0.0049,
        'Recall@10_mean'  : 0.8331,  'Recall@10_std'   : 0.0072,
        'F1@10_mean'      : 0.2515,  'F1@10_std'       : 0.0056,
        'NDCG@10_mean'    : 0.8071,  'NDCG@10_std'     : 0.0065,
        'RMSE_mean'       : 0.9905,  'RMSE_std'        : 0.0230,
        'MAE_mean'        : 0.7866,  'MAE_std'         : 0.0325,
    }
    summary_rows.append(tf_data)

    df_summary = pd.DataFrame(summary_rows).sort_values('Model')
    path_summary = os.path.join(OUT_DIR, 'reproducibility_summary.csv')
    df_summary.to_csv(path_summary, index=False)

    # ─── STAMPA RIEPILOGO ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("📊 RIEPILOGO RIPETIBILITÀ (media ± std su 3 seed)")
    print(f"{'='*70}")
    print(f"\n  {'Modello':<22} {'NDCG@10':>16} {'RMSE':>14} {'Recall@10':>16}")
    print(f"  {'-'*68}")
    for _, row in df_summary.iterrows():
        nd  = f"{row['NDCG@10_mean']:.4f} ±{row['NDCG@10_std']:.4f}"
        rm  = f"{row['RMSE_mean']:.4f} ±{row['RMSE_std']:.4f}"
        rec = f"{row['Recall@10_mean']:.4f} ±{row['Recall@10_std']:.4f}"
        print(f"  {row['Model']:<22} {nd:>16} {rm:>14} {rec:>16}")

    print(f"\n✅ Grezzi   → {path_raw}")
    print(f"✅ Summary  → {path_summary}")
    return df_summary

if __name__ == '__main__':
    main()
