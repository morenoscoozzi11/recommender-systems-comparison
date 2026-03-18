"""
SCRIPT 6: CONFRONTO MULTI-DIMENSIONALE
Esegue tutti e 5 i modelli su 3 dimensioni del dataset (10k, 40k, 100k righe)
e produce una tabella comparativa completa.

OUTPUT:
  comparison_multidim.csv  — tabella con tutti i risultati
  
PREREQUISITI:
  - amazon_cf_10k.csv, amazon_cf_40k.csv, amazon_cf_100k.csv
  - amazon_with_text_10k.csv, amazon_with_text_40k.csv, amazon_with_text_100k.csv
  (generati da preprocess_amazon.py)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── CONFIGURAZIONE ──────────────────────────────────────────────────────────
SIZES = ['10k', '40k', '100k']

CONFIG = {
    'RANDOM_STATE'         : 42,
    'TEST_SIZE'            : 0.2,
    'K'                    : 10,
    'THRESHOLD'            : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # Modello 1 — TF-IDF KNN
    'TFIDF_MAX_FEATURES'   : 5000,
    'KNN_N_NEIGHBORS'      : 20,
    # Modello 2 — Doc2Vec
    'D2V_VECTOR_SIZE'      : 100,
    'D2V_EPOCHS'           : 40,
    # Modello 3 — NMF
    'NMF_N_COMPONENTS'     : 20,
    'NMF_MAX_ITER'         : 300,
    # Modello 4 — Hybrid SVD+TFIDF
    'SVD_N_COMPONENTS'     : 20,
    'HYBRID_ALPHA'         : 0.5,
    'HYBRID_TFIDF_FEATURES': 1000,
    # Modello 5 — TensorFlow
    'TF_EMBEDDING_DIM'     : 32,
    'TF_TEXT_TOKENS'       : 1000,
    'TF_BATCH_SIZE'        : 32,
    'TF_EPOCHS'            : 30,
    'TF_LR'                : 0.005,
}

# ─── METRICHE CONDIVISE ──────────────────────────────────────────────────────

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
        dcg  = sum(1/np.log2(i+2) for i,(_, t) in enumerate(ratings[:k]) if t >= threshold)
        nrel = sum(1 for _,t in ratings if t >= threshold)
        idcg = sum(1/np.log2(i+2) for i in range(min(k, nrel)))
        ndcgs[uid] = dcg/idcg if idcg > 0 else 0.0
    return ndcgs

def compute_metrics(all_predictions, rmse_true, rmse_pred, k, threshold):
    prec, rec, f1 = precision_recall_at_k(all_predictions, k, threshold)
    ndcg          = ndcg_at_k(all_predictions, k, threshold)
    rmse = float(np.sqrt(mean_squared_error(rmse_true, rmse_pred)))
    mae  = float(np.mean(np.abs(np.array(rmse_true) - np.array(rmse_pred))))
    return {
        'Precision@10': round(np.mean(list(prec.values())), 4),
        'Recall@10'   : round(np.mean(list(rec.values())),  4),
        'F1@10'       : round(np.mean(list(f1.values())),   4),
        'NDCG@10'     : round(np.mean(list(ndcg.values())), 4),
        'RMSE'        : round(rmse, 4),
        'MAE'         : round(mae,  4),
    }

def load_and_filter(path_text=None, path_cf=None):
    """Carica e filtra il dataset. Ritorna df pronto per il training."""
    if path_text:
        df = pd.read_csv(path_text)
        df['reviewText'] = df['reviewText'].fillna('').astype(str)
    else:
        df = pd.read_csv(path_cf)
        df['reviewText'] = ''
    df = df.drop_duplicates(subset=['userID','itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]
    agg = {'rating': 'mean', 'reviewText': 'first'} if path_text else {'rating': 'mean'}
    df = df.groupby(['userID','itemID']).agg(agg).reset_index()
    if 'reviewText' not in df.columns:
        df['reviewText'] = ''
    return df

# ─── MODELLO 1: TF-IDF + KNN ─────────────────────────────────────────────────

def run_sklearn(df):
    t0 = time.perf_counter()
    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])
    item_list   = list(train['itemID'].unique())
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    item_text   = train.groupby('itemID')['reviewText'].first().to_dict()
    texts       = [item_text.get(it,'') for it in item_list]

    vec = TfidfVectorizer(stop_words='english',
                          max_features=CONFIG['TFIDF_MAX_FEATURES'])
    tfidf = vec.fit_transform(texts)

    n_nb = min(CONFIG['KNN_N_NEIGHBORS'], len(item_list)-1)
    knn  = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_nb)
    knn.fit(tfidf)

    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        rated = train[train['userID']==uid][['itemID','rating']]
        if rated.empty:
            for _, row in group.iterrows():
                all_preds.append((uid, row['itemID'], row['rating'], global_mean, 0))
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
        pred = np.where(mask, scores/weights, global_mean)
        pred = np.clip(pred, 1.0, 5.0)

        true_map = dict(zip(group['itemID'], group['rating']))
        top_k_idx = np.argsort(-pred)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it, 0.0), pred[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = pred[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp, CONFIG['K'], CONFIG['THRESHOLD'])
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    m['Num_Predictions']  = len(all_preds)
    return m

# ─── MODELLO 2: DOC2VEC ───────────────────────────────────────────────────────

def run_gensim(df):
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        import gensim
    except ImportError:
        return None

    t0 = time.perf_counter()
    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])
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
                    workers=1, seed=CONFIG['RANDOM_STATE'])
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

    vecs = np.array([model.dv[str(it)] for it in item_list])
    sim_matrix = cosine_similarity(vecs)

    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        rated = train[train['userID']==uid][['itemID','rating']]
        true_map = dict(zip(group['itemID'], group['rating']))

        if rated.empty:
            for _, row in group.iterrows():
                all_preds.append((uid, row['itemID'], row['rating'], global_mean, 0))
                rt.append(row['rating']); rp.append(global_mean)
            continue

        w_sum = np.zeros(len(item_list))
        w_tot = np.zeros(len(item_list))
        for _, r in rated.iterrows():
            if r['itemID'] not in item_to_idx: continue
            j = item_to_idx[r['itemID']]
            w_sum += sim_matrix[:, j] * r['rating']
            w_tot += sim_matrix[:, j]

        pred = np.where(w_tot > 0, w_sum/w_tot, global_mean)
        pred = np.clip(pred, 1.0, 5.0)

        top_k_idx = np.argsort(-pred)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it, 0.0), pred[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = pred[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp, CONFIG['K'], CONFIG['THRESHOLD'])
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    m['Num_Predictions']  = len(all_preds)
    return m

# ─── MODELLO 3: CF NMF ────────────────────────────────────────────────────────

def run_nmf(df):
    t0 = time.perf_counter()
    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])
    matrix = train.pivot_table(index='userID', columns='itemID',
                                values='rating', fill_value=0)
    user_list   = list(matrix.index)
    item_list   = list(matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    R = matrix.values.astype(np.float64)

    n_comp = min(CONFIG['NMF_N_COMPONENTS'], min(R.shape)-1)
    nmf = NMF(n_components=n_comp, max_iter=CONFIG['NMF_MAX_ITER'],
              random_state=CONFIG['RANDOM_STATE'], init='nndsvda')
    W = nmf.fit_transform(R)
    H = nmf.components_
    R_approx = np.clip(W @ H, 1.0, 5.0)

    global_mean = train['rating'].mean()
    all_preds, rt, rp = [], [], []

    for uid, group in test.groupby('userID'):
        true_map = dict(zip(group['itemID'], group['rating']))
        if uid not in user_to_idx:
            for _, row in group.iterrows():
                all_preds.append((uid, row['itemID'], row['rating'], global_mean, 0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        u = user_to_idx[uid]
        scores = R_approx[u, :]
        top_k_idx = np.argsort(-scores)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it, 0.0), scores[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = scores[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp, CONFIG['K'], CONFIG['THRESHOLD'])
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    m['Num_Predictions']  = len(all_preds)
    return m

# ─── MODELLO 4: HYBRID SVD+TF-IDF ────────────────────────────────────────────

def run_hybrid(df):
    t0 = time.perf_counter()
    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])
    matrix = train.pivot_table(index='userID', columns='itemID',
                                values='rating', fill_value=0)
    user_list   = list(matrix.index)
    item_list   = list(matrix.columns)
    user_to_idx = {u: i for i, u in enumerate(user_list)}
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    R = matrix.values.astype(np.float64)

    n_comp = min(CONFIG['SVD_N_COMPONENTS'], min(R.shape)-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=CONFIG['RANDOM_STATE'])
    U   = svd.fit_transform(R)
    Vt  = svd.components_
    R_cf = np.clip(U @ Vt, 1.0, 5.0)

    item_text_df = (train[['itemID','reviewText']].drop_duplicates('itemID')
                    .set_index('itemID').reindex(item_list).fillna(''))
    vec  = TfidfVectorizer(stop_words='english',
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
                all_preds.append((uid, row['itemID'], row['rating'], global_mean, 0))
                rt.append(row['rating']); rp.append(global_mean)
            continue
        u = user_to_idx[uid]
        cf_scores  = R_cf[u, :]
        rated_row  = matrix.loc[uid]
        rated_items = rated_row[rated_row > 0]

        w_sum = np.zeros(len(item_list))
        w_tot = np.zeros(len(item_list))
        for ri, rv in rated_items.items():
            if ri not in item_to_idx: continue
            j = item_to_idx[ri]
            w_sum += sim[:, j] * rv
            w_tot += sim[:, j]
        cbf_scores = np.where(w_tot > 0, np.clip(w_sum/w_tot,1,5), global_mean)
        hybrid     = alpha * cf_scores + (1-alpha) * cbf_scores

        top_k_idx = np.argsort(-hybrid)[:CONFIG['K']]
        for rank, i in enumerate(top_k_idx):
            it = item_list[i]
            all_preds.append((uid, it, true_map.get(it, 0.0), hybrid[i], rank))
        for _, row in group.iterrows():
            iid = row['itemID']
            p = hybrid[item_to_idx[iid]] if iid in item_to_idx else global_mean
            rt.append(row['rating']); rp.append(p)

    m = compute_metrics(all_preds, rt, rp, CONFIG['K'], CONFIG['THRESHOLD'])
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    m['Num_Predictions']  = len(all_preds)
    return m

# ─── MODELLO 5: TENSORFLOW ───────────────────────────────────────────────────

def run_tensorflow(df):
    try:
        import tensorflow as tf
        from tensorflow import keras
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.random.set_seed(CONFIG['RANDOM_STATE'])
    except ImportError:
        return None

    t0 = time.perf_counter()
    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])

    user_enc = LabelEncoder(); item_enc = LabelEncoder()
    user_enc.fit(train['userID']); item_enc.fit(train['itemID'])
    n_users = len(user_enc.classes_); n_items = len(item_enc.classes_)

    mask = (test['userID'].isin(user_enc.classes_) &
            test['itemID'].isin(item_enc.classes_))
    test = test[mask].copy()
    if test.empty: return None

    train_u = user_enc.transform(train['userID']).astype(np.int32)
    train_i = item_enc.transform(train['itemID']).astype(np.int32)
    test_u  = user_enc.transform(test['userID']).astype(np.int32)
    test_i  = item_enc.transform(test['itemID']).astype(np.int32)

    vec = TfidfVectorizer(max_features=CONFIG['TF_TEXT_TOKENS'], stop_words='english')
    vec.fit(train['reviewText'].values)
    item_map = (train[['itemID','reviewText']].drop_duplicates('itemID')
                .set_index('itemID')['reviewText'].to_dict())
    item_tfidf = np.zeros((n_items, CONFIG['TF_TEXT_TOKENS']), dtype=np.float32)
    for istr, idx in zip(item_enc.classes_, range(n_items)):
        item_tfidf[idx] = vec.transform([item_map.get(istr,'')]).toarray()[0]

    train_text = item_tfidf[train_i]
    test_text  = item_tfidf[test_i]

    emb = CONFIG['TF_EMBEDDING_DIM']
    tok = CONFIG['TF_TEXT_TOKENS']
    inp_u = keras.Input(shape=(1,), dtype='int32')
    inp_i = keras.Input(shape=(1,), dtype='int32')
    inp_t = keras.Input(shape=(tok,))
    ue = keras.layers.Flatten()(keras.layers.Embedding(n_users, emb)(inp_u))
    ie = keras.layers.Flatten()(keras.layers.Embedding(n_items, emb)(inp_i))
    tp = keras.layers.Dense(emb, activation='relu')(inp_t)
    x  = keras.layers.Concatenate()([ue, ie, tp])
    x  = keras.layers.Dropout(0.2)(keras.layers.Dense(128, activation='relu')(x))
    x  = keras.layers.Dropout(0.1)(keras.layers.Dense(64,  activation='relu')(x))
    out = keras.layers.Lambda(lambda z: z*4.0+1.0)(
          keras.layers.Dense(1, activation='sigmoid')(x))
    model = keras.Model(inputs=[inp_u, inp_i, inp_t], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(CONFIG['TF_LR']), loss='mse')

    model.fit([train_u, train_i, train_text],
              train['rating'].values.astype(np.float32),
              batch_size=CONFIG['TF_BATCH_SIZE'], epochs=CONFIG['TF_EPOCHS'],
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    preds = model.predict([test_u, test_i, test_text], verbose=0).squeeze()
    test = test.copy(); test['pred'] = preds

    all_preds, rt, rp = [], [], []
    for uid, group in test.groupby('userID'):
        gs = group.sort_values('pred', ascending=False)
        for _, row in gs.head(CONFIG['K']).iterrows():
            all_preds.append((uid, row['itemID'], row['rating'], row['pred'], 0))
        for _, row in group.iterrows():
            rt.append(row['rating']); rp.append(row['pred'])

    m = compute_metrics(all_preds, rt, rp, CONFIG['K'], CONFIG['THRESHOLD'])
    m['Execution_Time_s'] = round(time.perf_counter()-t0, 2)
    m['Num_Predictions']  = len(all_preds)
    return m

# ─── MAIN ─────────────────────────────────────────────────────────────────────

MODELS = [
    ('1 - Sklearn TF-IDF',    run_sklearn,    'text'),
    ('2 - Gensim Doc2Vec',    run_gensim,     'text'),
    ('3 - CF NMF',            run_nmf,        'cf'),
    ('4 - Hybrid CF+CBF',     run_hybrid,     'text'),
    ('5 - TensorFlow DL',     run_tensorflow, 'text'),
]

def main():
    print("=" * 80)
    print("📊  CONFRONTO MULTI-DIMENSIONALE — 5 Modelli × 3 Dimensioni Dataset")
    print("=" * 80)

    all_results = []

    for size in SIZES:
        path_text = os.path.join(BASE_DIR, f'amazon_with_text_{size}.csv')
        path_cf   = os.path.join(BASE_DIR, f'amazon_cf_{size}.csv')

        if not os.path.exists(path_text) or not os.path.exists(path_cf):
            print(f"\n⚠️  File mancanti per {size} — esegui prima preprocess_amazon.py")
            continue

        print(f"\n{'='*80}")
        print(f"📁  DATASET: {size.upper()}")
        print(f"{'='*80}")

        # Info dataset dopo filtro
        df_info = load_and_filter(path_text=path_text)
        print(f"   Dopo filtro min5: {len(df_info):,} righe | "
              f"{df_info['userID'].nunique()} utenti | "
              f"{df_info['itemID'].nunique()} item")

        for model_name, model_fn, dtype in MODELS:
            print(f"\n   ▶  {model_name}...", end='', flush=True)

            try:
                df = load_and_filter(
                    path_text=path_text if dtype == 'text' else None,
                    path_cf=path_cf     if dtype == 'cf'   else None
                )
                result = model_fn(df)

                if result is None:
                    print(" ⚠️  libreria non disponibile")
                    continue

                result['Model']   = model_name
                result['Dataset'] = size
                all_results.append(result)

                print(f" ✓  P@10={result['Precision@10']:.4f} | "
                      f"R@10={result['Recall@10']:.4f} | "
                      f"NDCG={result['NDCG@10']:.4f} | "
                      f"RMSE={result['RMSE']:.4f} | "
                      f"t={result['Execution_Time_s']}s")

            except Exception as e:
                print(f" ❌  Errore: {e}")

    if not all_results:
        print("\n❌  Nessun risultato prodotto.")
        return

    # ─── TABELLA COMPARATIVA ─────────────────────────────────────────────────
    cols = ['Dataset','Model','Precision@10','Recall@10','F1@10',
            'NDCG@10','RMSE','MAE','Execution_Time_s','Num_Predictions']
    df_res = pd.DataFrame(all_results)[cols]

    # Ordina per dimensione e modello
    size_order = {s: i for i, s in enumerate(SIZES)}
    df_res['_s'] = df_res['Dataset'].map(size_order)
    df_res = df_res.sort_values(['_s','Model']).drop(columns='_s')

    out_path = os.path.join(BASE_DIR, 'comparison_multidim.csv')
    df_res.to_csv(out_path, index=False)

    print(f"\n\n{'='*80}")
    print("🏆  TABELLA COMPARATIVA FINALE")
    print(f"{'='*80}")
    print(df_res.to_string(index=False))
    print(f"\n✅  Risultati salvati: {out_path}")

    # ─── RIEPILOGO MIGLIORI PER METRICA × DIMENSIONE ─────────────────────────
    print(f"\n{'='*80}")
    print("⭐  MIGLIORI PER METRICA E DIMENSIONE")
    print(f"{'='*80}")
    for size in SIZES:
        sub = df_res[df_res['Dataset'] == size]
        if sub.empty: continue
        print(f"\n  📁 {size.upper()}:")
        for metric in ['Precision@10','Recall@10','NDCG@10','RMSE']:
            if metric == 'RMSE':
                best = sub.loc[sub[metric].idxmin()]
            else:
                best = sub.loc[sub[metric].idxmax()]
            print(f"    {metric:<14}: {best['Model']}  ({best[metric]:.4f})")

if __name__ == '__main__':
    main()
