"""
PREPROCESSING — Amazon Movies & TV Dataset
Genera i CSV per 3 dimensioni del dataset: 10k, 40k, 100k righe.

FIXES rispetto alla versione originale:
  1. Salvataggio nella stessa cartella dello script (non in data/)
  2. rating_binary usa soglia >= 4.0 (coerente con le metriche dei modelli)
  3. Rimossi calcoli inutili (COO matrix, TF-IDF) che non venivano salvati
  4. Output separato per ogni dimensione: amazon_cf_10k.csv, ecc.

USO:
  python3 preprocess_amazon.py

Il file movies.txt.gz deve trovarsi nella stessa cartella dello script.
"""

import os
import gzip
import pandas as pd

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, 'movies.txt.gz')

SIZES = [10_000, 40_000, 100_000]   # righe da estrarre per ogni variante

# Soglia coerente con i modelli: rating >= 4.0 → rilevante
BINARY_THRESHOLD = 4.0

# ─── PARSER ──────────────────────────────────────────────────────────────────

def parse_text_file(path: str, limit: int) -> pd.DataFrame:
    """
    Legge il file .txt.gz formato Amazon (coppie chiave:valore separate da righe vuote).
    Restituisce un DataFrame con le prime `limit` recensioni.
    """
    data  = []
    entry = {}

    print(f"   📖 Lettura file (limite: {limit:,} righe)...", end='', flush=True)
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if entry:
                    data.append(entry)
                    entry = {}
                    if len(data) >= limit:
                        break
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                entry[key.strip()] = value.strip()

    if entry and len(data) < limit:
        data.append(entry)

    df = pd.DataFrame(data)
    print(f" ✓  ({len(df):,} record grezzi)")
    return df


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess(df_raw: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partire dal DataFrame grezzo:
      - Estrae le colonne necessarie
      - Pulisce i tipi
      - Crea rating_binary con soglia 4.0
      - Restituisce (df_cf, df_text)
    """

    # --- Collaborative Filtering (senza testo) ---
    needed_cf = ['review/userId', 'product/productId', 'review/score']
    df_cf = df_raw[needed_cf].dropna().copy()
    df_cf.columns = ['userID', 'itemID', 'rating']
    df_cf['rating'] = pd.to_numeric(df_cf['rating'], errors='coerce')
    df_cf = df_cf.dropna(subset=['rating'])
    df_cf['rating_binary'] = (df_cf['rating'] >= BINARY_THRESHOLD).astype(int)

    # --- Con testo (per CBF e Hybrid) ---
    needed_text = ['review/userId', 'product/productId', 'review/score', 'review/text']
    available = [c for c in needed_text if c in df_raw.columns]
    df_text = df_raw[available].dropna().copy()
    df_text.columns = ['userID', 'itemID', 'rating'] + (['reviewText'] if 'review/text' in available else [])
    df_text['rating'] = pd.to_numeric(df_text['rating'], errors='coerce')
    df_text = df_text.dropna(subset=['rating'])
    if 'reviewText' not in df_text.columns:
        df_text['reviewText'] = ''

    # Stats
    print(f"\n   📊 [{label}] Statistiche:")
    print(f"      CF   → {len(df_cf):>6,} righe | {df_cf['userID'].nunique():>5,} utenti | {df_cf['itemID'].nunique():>4,} item")
    print(f"      Text → {len(df_text):>6,} righe | {df_text['userID'].nunique():>5,} utenti | {df_text['itemID'].nunique():>4,} item")
    print(f"      Rating medio: {df_cf['rating'].mean():.2f} | rating_binary >= {BINARY_THRESHOLD}: {df_cf['rating_binary'].mean()*100:.1f}%")

    return df_cf, df_text


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🔧  PREPROCESSING — Amazon Movies & TV")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"\n❌  File non trovato: {DATA_FILE}")
        print("   Assicurati che 'movies.txt.gz' sia nella stessa cartella dello script.")
        return

    # Legge il massimo necessario una volta sola
    max_limit = max(SIZES)
    df_raw = parse_text_file(DATA_FILE, limit=max_limit)

    # Genera un CSV per ogni dimensione
    generated = []
    for size in SIZES:
        label  = f"{size // 1000}k"
        subset = df_raw.head(size)

        df_cf, df_text = preprocess(subset, label)

        path_cf   = os.path.join(BASE_DIR, f'amazon_cf_{label}.csv')
        path_text = os.path.join(BASE_DIR, f'amazon_with_text_{label}.csv')

        df_cf.to_csv(path_cf,   index=False)
        df_text.to_csv(path_text, index=False)

        print(f"      💾 Salvati: amazon_cf_{label}.csv | amazon_with_text_{label}.csv")
        generated.append((label, len(df_cf), df_cf['userID'].nunique(), df_cf['itemID'].nunique()))

    # Riepilogo finale
    print("\n" + "=" * 70)
    print("✅  RIEPILOGO FILE GENERATI")
    print("=" * 70)
    print(f"{'Dimensione':<12} {'Righe CF':>10} {'Utenti':>8} {'Item':>6}")
    print("-" * 40)
    for label, rows, users, items in generated:
        print(f"{label:<12} {rows:>10,} {users:>8,} {items:>6,}")
    print()
    print(f"📁 Tutti i file salvati in: {BASE_DIR}")


if __name__ == '__main__':
    main()
