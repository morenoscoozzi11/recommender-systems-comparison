"""
MODELLO 5: TENSORFLOW / KERAS (Hybrid — Deep Learning)
Approccio: User embedding + Item embedding + TF-IDF text features → rating prediction
Valutazione: Train/Test split 80/20, metriche standardizzate

NOTA: Usa solo TensorFlow/Keras puro, SENZA tensorflow_recommenders,
      per compatibilità con Keras 3.x.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow non disponibile. "
          "Installa con: pip install tensorflow --break-system-packages")

# ========== CONFIGURAZIONE ==========
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'DATASET_PATH'        : os.path.join(DATA_DIR, 'amazon_with_text_features.csv'),
    'OUTPUT_PATH'         : os.path.join(OUT_DIR,  'results_tensorflow.csv'),
    'RANDOM_STATE'        : 42,
    'TEST_SIZE'           : 0.2,
    'K'                   : 10,
    'THRESHOLD'           : 4.0,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    # Keras model
    'EMBEDDING_DIM'  : 32,
    'TEXT_MAX_TOKENS': 1000,
    'BATCH_SIZE'     : 32,
    'N_EPOCHS'       : 30,
    'LEARNING_RATE'  : 0.005,
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
    if not TF_AVAILABLE:
        print("❌ TensorFlow non disponibile.")
        return

    tf.random.set_seed(CONFIG['RANDOM_STATE'])
    np.random.seed(CONFIG['RANDOM_STATE'])

    # Sopprime log TF non essenziali
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("=" * 80)
    print("🔹 MODELLO 5: TENSORFLOW / KERAS (Hybrid — Deep Learning)")
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

    # ----- encoding user e item con interi -----
    print(f"\n🔧 Encoding utenti e item...")
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    user_enc.fit(train_df['userID'])
    item_enc.fit(train_df['itemID'])

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)
    print(f"   ✓ {n_users} utenti, {n_items} item")

    # filtra test: solo utenti e item visti nel training
    mask = (test_df['userID'].isin(user_enc.classes_) &
            test_df['itemID'].isin(item_enc.classes_))
    test_df = test_df[mask].copy()
    print(f"   ✓ Test dopo filtro cold-start: {len(test_df)} righe")

    train_user_idx = user_enc.transform(train_df['userID']).astype(np.int32)
    train_item_idx = item_enc.transform(train_df['itemID']).astype(np.int32)
    test_user_idx  = user_enc.transform(test_df['userID']).astype(np.int32)
    test_item_idx  = item_enc.transform(test_df['itemID']).astype(np.int32)

    # ----- TF-IDF features -----
    print(f"\n📝 TF-IDF text features (max_tokens={CONFIG['TEXT_MAX_TOKENS']})...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['TEXT_MAX_TOKENS'],
        stop_words='english'
    )
    vectorizer.fit(train_df['reviewText'].values)

    # features per ogni item (media delle review dello stesso item nel training)
    item_text_map = (
        train_df[['itemID', 'reviewText']]
        .drop_duplicates('itemID')
        .set_index('itemID')['reviewText']
        .to_dict()
    )
    item_tfidf = np.zeros((n_items, CONFIG['TEXT_MAX_TOKENS']), dtype=np.float32)
    for item_str, idx in zip(item_enc.classes_, range(n_items)):
        text = item_text_map.get(item_str, '')
        item_tfidf[idx] = vectorizer.transform([text]).toarray()[0]

    train_text = item_tfidf[train_item_idx]
    test_text  = item_tfidf[test_item_idx]
    print(f"   ✓ TF-IDF matrix: {item_tfidf.shape}")

    # ----- costruzione modello Keras -----
    print(f"\n🤖 Building modello Keras (embedding_dim={CONFIG['EMBEDDING_DIM']})...")

    # Input
    input_user = keras.Input(shape=(1,),   name='user_id',   dtype='int32')
    input_item = keras.Input(shape=(1,),   name='item_id',   dtype='int32')
    input_text = keras.Input(shape=(CONFIG['TEXT_MAX_TOKENS'],), name='text_feat')

    # User embedding
    user_emb = keras.layers.Embedding(
        input_dim=n_users, output_dim=CONFIG['EMBEDDING_DIM'], name='user_emb'
    )(input_user)
    user_emb = keras.layers.Flatten()(user_emb)

    # Item embedding
    item_emb = keras.layers.Embedding(
        input_dim=n_items, output_dim=CONFIG['EMBEDDING_DIM'], name='item_emb'
    )(input_item)
    item_emb = keras.layers.Flatten()(item_emb)

    # Text projection
    text_proj = keras.layers.Dense(CONFIG['EMBEDDING_DIM'], activation='relu',
                                   name='text_proj')(input_text)

    # Concatena tutto
    concat = keras.layers.Concatenate()([user_emb, item_emb, text_proj])
    x = keras.layers.Dense(128, activation='relu')(concat)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='rating_out')(x)
    # scala da [0,1] a [1,5]
    output_scaled = keras.layers.Lambda(
        lambda x: x * 4.0 + 1.0, name='scale'
    )(output)

    model = keras.Model(
        inputs=[input_user, input_item, input_text],
        outputs=output_scaled
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='mse',
        metrics=['mae']
    )
    print(f"   ✓ Modello costruito ({model.count_params()} parametri)")

    # ----- training -----
    print(f"\n⚡ Training ({CONFIG['N_EPOCHS']} epochs, batch_size={CONFIG['BATCH_SIZE']})...")

    train_ratings = train_df['rating'].values.astype(np.float32)
    test_ratings  = test_df['rating'].values.astype(np.float32)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        x=[train_user_idx, train_item_idx, train_text],
        y=train_ratings,
        batch_size=CONFIG['BATCH_SIZE'],
        epochs=CONFIG['N_EPOCHS'],
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    epochs_run = len(history.history['loss'])
    best_val   = min(history.history['val_loss'])
    print(f"   ✓ Training completato — {epochs_run} epoch, best val_loss: {best_val:.4f}")

    # ----- predizioni sul test set -----
    print(f"\n🎯 Generazione predizioni sul test set...")

    raw_preds = model.predict(
        [test_user_idx, test_item_idx, test_text],
        batch_size=256, verbose=0
    ).squeeze()

    # per le ranking metrics genera top-K per utente
    test_df = test_df.copy()
    test_df['pred'] = raw_preds

    all_predictions = []
    rmse_true, rmse_pred = [], []

    for uid, group in test_df.groupby('userID'):
        group_sorted = group.sort_values('pred', ascending=False)
        top_k = group_sorted.head(CONFIG['K'])
        for _, row in top_k.iterrows():
            all_predictions.append(
                (uid, row['itemID'], row['rating'], row['pred'], 0)
            )
        for _, row in group.iterrows():
            rmse_true.append(row['rating'])
            rmse_pred.append(row['pred'])

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
        'Model'           : 'TensorFlow (Hybrid-DL)',
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
