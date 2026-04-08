"""
SCRIPT 8: ANALISI DEGLI ERRORI — TENSORFLOW DL (dataset 40k)
Spiega il paradosso Precision@10 bassa / Recall@10 alta come artefatto
strutturale della valutazione, non come difetto del modello.

OUTPUT:
  analisi_errori_tf_40k.png

PREREQUISITI:
  - amazon_with_text_40k.csv  (generato da preprocess_amazon.py)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'         : os.path.join(BASE_DIR, 'amazon_with_text_40k.csv'),
    'OUTPUT_PATH'          : os.path.join(BASE_DIR, 'analisi_errori_tf_40k.png'),
    'RANDOM_STATE'         : 42,
    'TEST_SIZE'            : 0.2,
    'K'                    : 10,
    'THRESHOLD'            : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    'TF_EMBEDDING_DIM'     : 32,
    'TF_TEXT_TOKENS'       : 1000,
    'TF_BATCH_SIZE'        : 32,
    'TF_EPOCHS'            : 30,
    'TF_LR'                : 0.005,
}


def load_and_filter(path):
    df = pd.read_csv(path)
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    df = df.drop_duplicates(subset=['userID', 'itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]
    df = df.groupby(['userID', 'itemID']).agg(
        rating=('rating', 'mean'),
        reviewText=('reviewText', 'first')
    ).reset_index()
    return df


def run_tensorflow(df):
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(CONFIG['RANDOM_STATE'])
    except ImportError:
        print("TensorFlow non disponibile.")
        return None

    train, test = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                   random_state=CONFIG['RANDOM_STATE'])
    user_enc = LabelEncoder(); item_enc = LabelEncoder()
    user_enc.fit(train['userID']); item_enc.fit(train['itemID'])
    n_users = len(user_enc.classes_); n_items = len(item_enc.classes_)

    mask = (test['userID'].isin(user_enc.classes_) &
            test['itemID'].isin(item_enc.classes_))
    test = test[mask].copy()

    train_u = user_enc.transform(train['userID']).astype(np.int32)
    train_i = item_enc.transform(train['itemID']).astype(np.int32)
    test_u  = user_enc.transform(test['userID']).astype(np.int32)
    test_i  = item_enc.transform(test['itemID']).astype(np.int32)

    vec = TfidfVectorizer(max_features=CONFIG['TF_TEXT_TOKENS'],
                          stop_words='english')
    vec.fit(train['reviewText'].values)
    item_map = (train[['itemID', 'reviewText']].drop_duplicates('itemID')
                .set_index('itemID')['reviewText'].to_dict())
    item_tfidf = np.zeros((n_items, CONFIG['TF_TEXT_TOKENS']), dtype=np.float32)
    for istr, idx in zip(item_enc.classes_, range(n_items)):
        item_tfidf[idx] = vec.transform([item_map.get(istr, '')]).toarray()[0]

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
    x  = keras.layers.Dropout(0.1)(keras.layers.Dense(64, activation='relu')(x))
    out = keras.layers.Lambda(lambda z: z * 4.0 + 1.0)(
          keras.layers.Dense(1, activation='sigmoid')(x))
    model = keras.Model(inputs=[inp_u, inp_i, inp_t], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(CONFIG['TF_LR']), loss='mse')
    model.fit([train_u, train_i, train_text],
              train['rating'].values.astype(np.float32),
              batch_size=CONFIG['TF_BATCH_SIZE'],
              epochs=CONFIG['TF_EPOCHS'],
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=5, restore_best_weights=True)],
              verbose=0)

    preds = model.predict([test_u, test_i, test_text], verbose=0).squeeze()
    test = test.copy()
    test['pred'] = preds
    return test


def compute_per_user_metrics(test_df):
    results = []
    for uid, group in test_df.groupby('userID'):
        n_test = len(group)
        n_relevant = (group['rating'] >= CONFIG['THRESHOLD']).sum()
        top_k = group.sort_values('pred', ascending=False).head(CONFIG['K'])
        n_hits = (top_k['rating'] >= CONFIG['THRESHOLD']).sum()
        prec = n_hits / CONFIG['K']
        rec  = n_hits / n_relevant if n_relevant > 0 else 0.0
        results.append({'n_test_items': n_test, 'precision': prec, 'recall': rec})
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("📊  ANALISI ERRORI — TensorFlow DL (dataset 40k)")
    print("=" * 70)

    if not os.path.exists(CONFIG['DATASET_PATH']):
        print(f"❌  File non trovato: {CONFIG['DATASET_PATH']}")
        return

    print("\n📂 Caricamento dataset...")
    df = load_and_filter(CONFIG['DATASET_PATH'])
    print(f"   ✓ {len(df):,} righe | {df['userID'].nunique()} utenti | "
          f"{df['itemID'].nunique()} item")

    print("\n🤖 Training TensorFlow (seed=42)...")
    test_df = run_tensorflow(df)
    if test_df is None:
        return
    print(f"   ✓ Predizioni su {len(test_df)} righe")

    per_user  = compute_per_user_metrics(test_df)
    residuals = test_df['pred'] - test_df['rating']
    pct_single = (per_user['n_test_items'] == 1).mean() * 100

    print(f"\n   Utenti con 1 item nel test : {pct_single:.1f}%")
    print(f"   RMSE                        : {np.sqrt((residuals**2).mean()):.4f}")
    print(f"   Bias medio (pred − true)    : {residuals.mean():.4f}")

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Distribuzione n° item per utente nel test
    counts = per_user['n_test_items'].value_counts().sort_index()
    colors = ['tomato' if x == 1 else 'steelblue' for x in counts.index]
    ax1.bar(counts.index.astype(str), counts.values,
            color=colors, edgecolor='white', linewidth=0.5)
    ax1.bar([], [], color='tomato',
            label=f'1 solo item ({pct_single:.0f}% utenti)')
    ax1.set_xlabel('N° item nel test set per utente', fontsize=11)
    ax1.set_ylabel('N° utenti', fontsize=11)
    ax1.set_title('Distribuzione item per utente\n(test set)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

    # 2. Scatter rating reale vs score predetto
    sample = test_df.sample(min(800, len(test_df)), random_state=42)
    ax2.scatter(sample['rating'], sample['pred'],
                alpha=0.35, s=18, color='steelblue', edgecolors='none')
    ax2.plot([1, 5], [1, 5], 'r--', linewidth=1.2, label='Predizione perfetta')
    ax2.set_xlabel('Rating reale', fontsize=11)
    ax2.set_ylabel('Score predetto', fontsize=11)
    ax2.set_title('Rating reale vs Score predetto', fontsize=11, fontweight='bold')
    ax2.set_xlim(0.8, 5.2); ax2.set_ylim(0.8, 5.2)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # 3. Distribuzione residui
    ax3.hist(residuals, bins=40, color='steelblue',
             edgecolor='white', linewidth=0.4)
    ax3.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Zero bias')
    ax3.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Bias medio: {residuals.mean():.2f}')
    ax3.set_xlabel('Residuo (pred − rating reale)', fontsize=11)
    ax3.set_ylabel('Frequenza', fontsize=11)
    ax3.set_title('Distribuzione degli errori (residui)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

    # 4. Precision@10 vs Recall@10 per utente
    ax4.scatter(per_user['recall'], per_user['precision'],
                alpha=0.4, s=18, color='steelblue', edgecolors='none')
    ax4.axhline(1 / CONFIG['K'], color='tomato', linestyle='--', linewidth=1.2,
                label=f'Precision max con 1 item = {1/CONFIG["K"]:.2f}')
    ax4.set_xlabel('Recall@10', fontsize=11)
    ax4.set_ylabel('Precision@10', fontsize=11)
    ax4.set_title('Precision@10 vs Recall@10 per utente', fontsize=11, fontweight='bold')
    ax4.set_xlim(-0.05, 1.05); ax4.set_ylim(-0.02, 0.55)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    fig.suptitle('Analisi degli errori — TensorFlow DL (dataset 40k)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(CONFIG['OUTPUT_PATH'], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅  Grafico salvato: {CONFIG['OUTPUT_PATH']}")


if __name__ == '__main__':
    main()
