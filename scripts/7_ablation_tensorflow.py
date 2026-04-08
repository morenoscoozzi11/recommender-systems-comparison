"""
SCRIPT 7: ABLATION STUDY — Iperparametri TensorFlow DL
=======================================================
Studia l'impatto di 4 iperparametri chiave sul modello TensorFlow (Modello 5),
variando un parametro alla volta e tenendo gli altri fissi ai valori base.

Iperparametri studiati:
  - embedding_dim   : [16, 32*, 64]
  - learning_rate   : [0.001, 0.005*, 0.01]
  - batch_size      : [32*, 64, 128]
  - text_max_tokens : [500, 1000*, 2000]
  (* = valore base originale)

Strategia: Ablation Study (non grid search)
  - Un parametro alla volta, altri fissi
  - 3 ripetizioni per configurazione → media ± std
  - Dataset: amazon_with_text_40k.csv (bilanciamento velocità/stabilità)

OUTPUT:
  ablation_results.csv     — tutti i risultati per configurazione e seed
  ablation_summary.csv     — media e std per ogni configurazione
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── IMPORT TENSORFLOW ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow non disponibile.")
    exit(1)

# ─── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'amazon_with_text_40k.csv')
OUT_DIR      = BASE_DIR

# ─── CONFIGURAZIONE BASE (valori originali del Modello 5) ────────────────────
BASE_CONFIG = {
    'EMBEDDING_DIM'  : 32,
    'LEARNING_RATE'  : 0.005,
    'BATCH_SIZE'     : 32,
    'TEXT_MAX_TOKENS': 1000,
    # Parametri fissi — non oggetto di ablation
    'N_EPOCHS'       : 30,
    'K'              : 10,
    'THRESHOLD'      : 4.0,
    'TEST_SIZE'      : 0.2,
    'MIN_USER_INT'   : 5,
    'MIN_ITEM_INT'   : 5,
    'PATIENCE'       : 5,
}

# ─── VARIANTI DA TESTARE (ablation: un parametro alla volta) ─────────────────
ABLATION_PLAN = {
    'EMBEDDING_DIM'  : [16, 32, 64],
    'LEARNING_RATE'  : [0.001, 0.005, 0.01],
    'BATCH_SIZE'     : [32, 64, 128],
    'TEXT_MAX_TOKENS': [500, 1000, 2000],
}

# 3 seed diversi per ripetibilità statistica
SEEDS = [42, 123, 7]

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
        dcg  = sum(1/np.log2(i+2) for i,(_, t) in enumerate(ratings[:k]) if t >= threshold)
        nrel = sum(1 for _, t in ratings if t >= threshold)
        idcg = sum(1/np.log2(i+2) for i in range(min(k, nrel)))
        ndcgs[uid] = dcg/idcg if idcg > 0 else 0.0
    return ndcgs

# ─── CARICAMENTO E FILTRAGGIO DATASET ────────────────────────────────────────

def load_dataset(path, cfg):
    df = pd.read_csv(path)
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    df = df.drop_duplicates(subset=['userID','itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= cfg['MIN_USER_INT']].index)]
    df = df[df['itemID'].isin(ic[ic >= cfg['MIN_ITEM_INT']].index)]
    df = df.groupby(['userID','itemID']).agg(
        rating=('rating','mean'),
        reviewText=('reviewText','first')
    ).reset_index()
    return df

# ─── SINGOLO ESPERIMENTO ──────────────────────────────────────────────────────

def run_experiment(df, cfg, seed):
    """
    Esegue un singolo training+valutazione con la configurazione e il seed dati.
    Restituisce un dizionario con tutte le metriche.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()

    # Split
    train_df, test_df = train_test_split(
        df, test_size=cfg['TEST_SIZE'], random_state=seed
    )

    # Encoding
    user_enc = LabelEncoder(); item_enc = LabelEncoder()
    user_enc.fit(train_df['userID']); item_enc.fit(train_df['itemID'])
    n_users = len(user_enc.classes_); n_items = len(item_enc.classes_)

    mask = (test_df['userID'].isin(user_enc.classes_) &
            test_df['itemID'].isin(item_enc.classes_))
    test_df = test_df[mask].copy()
    if test_df.empty:
        return None

    train_u = user_enc.transform(train_df['userID']).astype(np.int32)
    train_i = item_enc.transform(train_df['itemID']).astype(np.int32)
    test_u  = user_enc.transform(test_df['userID']).astype(np.int32)
    test_i  = item_enc.transform(test_df['itemID']).astype(np.int32)

    # TF-IDF
    tok = cfg['TEXT_MAX_TOKENS']
    vec = TfidfVectorizer(max_features=tok, stop_words='english')
    vec.fit(train_df['reviewText'].values)
    item_map = (train_df[['itemID','reviewText']].drop_duplicates('itemID')
                .set_index('itemID')['reviewText'].to_dict())
    item_tfidf = np.zeros((n_items, tok), dtype=np.float32)
    for istr, idx in zip(item_enc.classes_, range(n_items)):
        item_tfidf[idx] = vec.transform([item_map.get(istr,'')]).toarray()[0]

    train_text = item_tfidf[train_i]
    test_text  = item_tfidf[test_i]

    # Modello Keras — identico al Modello 5 originale
    emb = cfg['EMBEDDING_DIM']
    inp_u = keras.Input(shape=(1,), dtype='int32')
    inp_i = keras.Input(shape=(1,), dtype='int32')
    inp_t = keras.Input(shape=(tok,))
    ue = keras.layers.Flatten()(keras.layers.Embedding(n_users, emb)(inp_u))
    ie = keras.layers.Flatten()(keras.layers.Embedding(n_items, emb)(inp_i))
    tp = keras.layers.Dense(emb, activation='relu')(inp_t)
    x  = keras.layers.Concatenate()([ue, ie, tp])
    x  = keras.layers.Dropout(0.2)(keras.layers.Dense(128, activation='relu')(x))
    x  = keras.layers.Dropout(0.1)(keras.layers.Dense(64,  activation='relu')(x))
    out = keras.layers.Lambda(lambda z: z * 4.0 + 1.0)(
          keras.layers.Dense(1, activation='sigmoid')(x))
    model = keras.Model(inputs=[inp_u, inp_i, inp_t], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg['LEARNING_RATE']),
        loss='mse'
    )

    # Training
    history = model.fit(
        [train_u, train_i, train_text],
        train_df['rating'].values.astype(np.float32),
        batch_size=cfg['BATCH_SIZE'],
        epochs=cfg['N_EPOCHS'],
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=cfg['PATIENCE'], restore_best_weights=True
        )],
        verbose=0
    )
    epochs_run = len(history.history['loss'])

    # Predizioni
    preds = model.predict([test_u, test_i, test_text], verbose=0).squeeze()
    test_df = test_df.copy(); test_df['pred'] = preds

    all_preds, rt, rp = [], [], []
    for uid, group in test_df.groupby('userID'):
        gs = group.sort_values('pred', ascending=False)
        for _, row in gs.head(cfg['K']).iterrows():
            all_preds.append((uid, row['itemID'], row['rating'], row['pred'], 0))
        for _, row in group.iterrows():
            rt.append(row['rating']); rp.append(row['pred'])

    if not all_preds:
        return None

    # Metriche
    prec, rec, f1 = precision_recall_at_k(all_preds, cfg['K'], cfg['THRESHOLD'])
    ndcg = ndcg_at_k(all_preds, cfg['K'], cfg['THRESHOLD'])
    rmse = float(np.sqrt(mean_squared_error(rt, rp)))
    mae  = float(np.mean(np.abs(np.array(rt) - np.array(rp))))

    keras.backend.clear_session()  # libera memoria tra un run e l'altro

    return {
        'Precision@10'    : round(np.mean(list(prec.values())), 4),
        'Recall@10'       : round(np.mean(list(rec.values())),  4),
        'F1@10'           : round(np.mean(list(f1.values())),   4),
        'NDCG@10'         : round(np.mean(list(ndcg.values())), 4),
        'RMSE'            : round(rmse, 4),
        'MAE'             : round(mae,  4),
        'Epochs_run'      : epochs_run,
        'Execution_Time_s': round(time.perf_counter() - t0, 2),
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("🔬 ABLATION STUDY — Iperparametri TensorFlow DL")
    print(f"   Dataset  : amazon_with_text_40k.csv")
    print(f"   Seed     : {SEEDS}  (media ± std su 3 run)")
    print(f"   Strategia: un parametro alla volta, altri fissi ai valori base")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset non trovato: {DATASET_PATH}")
        print("   Esegui prima: python3 scripts/preprocess_amazon.py")
        return

    print(f"\n📂 Caricamento dataset 40k...")
    df = load_dataset(DATASET_PATH, BASE_CONFIG)
    print(f"   ✓ {len(df):,} righe | "
          f"{df['userID'].nunique()} utenti | "
          f"{df['itemID'].nunique()} item")

    all_results = []
    total = sum(len(v) for v in ABLATION_PLAN.values()) * len(SEEDS)
    done  = 0

    for param_name, values in ABLATION_PLAN.items():
        print(f"\n{'='*80}")
        print(f"📊 Ablation su: {param_name}")
        print(f"   Valori testati : {values}")
        print(f"   Valore base    : {BASE_CONFIG[param_name]}")
        print(f"{'='*80}")

        for val in values:
            cfg = BASE_CONFIG.copy()
            cfg[param_name] = val
            is_baseline = (val == BASE_CONFIG[param_name])
            label = f"{param_name}={val}" + (" [BASE]" if is_baseline else "")
            print(f"\n   ▶  {label}")

            for seed in SEEDS:
                done += 1
                print(f"      Seed {seed} ({done}/{total})...", end='', flush=True)
                try:
                    result = run_experiment(df, cfg, seed)
                    if result is None:
                        print(" ⚠️  nessuna predizione")
                        continue
                    result['param_name']  = param_name
                    result['param_value'] = val
                    result['is_baseline'] = is_baseline
                    result['seed']        = seed
                    all_results.append(result)
                    print(f" ✓  NDCG={result['NDCG@10']:.4f} | "
                          f"Recall={result['Recall@10']:.4f} | "
                          f"RMSE={result['RMSE']:.4f} | "
                          f"t={result['Execution_Time_s']}s | "
                          f"ep={result['Epochs_run']}")
                except Exception as e:
                    print(f" ❌ Errore: {e}")

    if not all_results:
        print("\n❌ Nessun risultato prodotto.")
        return

    # ─── SALVATAGGIO RISULTATI GREZZI ─────────────────────────────────────────
    cols_raw = ['param_name','param_value','is_baseline','seed',
                'Precision@10','Recall@10','F1@10','NDCG@10',
                'RMSE','MAE','Epochs_run','Execution_Time_s']
    df_raw = pd.DataFrame(all_results)[cols_raw]
    path_raw = os.path.join(OUT_DIR, 'ablation_results.csv')
    df_raw.to_csv(path_raw, index=False)

    # ─── SUMMARY: MEDIA ± STD ─────────────────────────────────────────────────
    metrics = ['Precision@10','Recall@10','F1@10','NDCG@10','RMSE','MAE','Epochs_run']
    summary_rows = []
    for (pname, pval), grp in df_raw.groupby(['param_name','param_value']):
        row = {
            'param_name' : pname,
            'param_value': pval,
            'is_baseline': bool(grp['is_baseline'].iloc[0]),
            'n_runs'     : len(grp),
        }
        for m in metrics:
            row[f'{m}_mean'] = round(grp[m].mean(), 4)
            row[f'{m}_std']  = round(grp[m].std(),  4)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(['param_name','param_value'])
    path_summary = os.path.join(OUT_DIR, 'ablation_summary.csv')
    df_summary.to_csv(path_summary, index=False)

    # ─── STAMPA RIEPILOGO ─────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("📊 RIEPILOGO ABLATION STUDY  (media ± std su 3 seed)")
    print(f"{'='*80}")

    for pname in ABLATION_PLAN.keys():
        sub = df_summary[df_summary['param_name'] == pname]
        print(f"\n  📌 {pname}")
        print(f"  {'Valore':<14} {'NDCG@10':>16} {'Recall@10':>16} {'RMSE':>14}")
        print(f"  {'-'*62}")
        for _, row in sub.iterrows():
            marker  = '  ◀ BASE' if row['is_baseline'] else ''
            nd_s    = f"{row['NDCG@10_mean']:.4f} ±{row['NDCG@10_std']:.4f}"
            rec_s   = f"{row['Recall@10_mean']:.4f} ±{row['Recall@10_std']:.4f}"
            rmse_s  = f"{row['RMSE_mean']:.4f} ±{row['RMSE_std']:.4f}"
            print(f"  {str(row['param_value']):<14} {nd_s:>16} {rec_s:>16} {rmse_s:>14}{marker}")

    # ─── CONFIGURAZIONE OTTIMALE ───────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("🏆 CONFIGURAZIONE OTTIMALE SUGGERITA (massimo NDCG@10)")
    print(f"{'='*80}")

    for pname in ABLATION_PLAN.keys():
        sub      = df_summary[df_summary['param_name'] == pname]
        best_row = sub.loc[sub['NDCG@10_mean'].idxmax()]
        base_val = BASE_CONFIG[pname]
        change   = "" if best_row['param_value'] == base_val \
                   else f"  ← suggerisce di cambiare da {base_val}"
        print(f"   {pname:<22}: {best_row['param_value']}{change}")
        print(f"   {'':22}  NDCG = {best_row['NDCG@10_mean']:.4f} ± {best_row['NDCG@10_std']:.4f}")

    print(f"\n✅ Risultati grezzi  → {path_raw}")
    print(f"✅ Summary (media±std) → {path_summary}")

    return df_summary


if __name__ == '__main__':
    main()
