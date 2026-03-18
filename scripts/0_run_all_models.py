"""
SCRIPT MASTER: Esecuzione di TUTTI i modelli di raccomandazione
Raccoglie i risultati in una tabella comparativa finale

Uso:
    python3 0_run_all_models.py

Gli script 1-5 devono trovarsi nella STESSA directory di questo file.
I risultati CSV vengono salvati nella stessa cartella degli script (OUT_DIR).
"""

import os
import sys
import subprocess
import time
from datetime import datetime

import pandas as pd

# ========== CONFIGURAZIONE ==========
# Directory in cui si trovano gli script 1-5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory di output (deve coincidere con OUT_DIR nei singoli script)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ('1_cbf_sklearn.py',    'Scikit-Learn (CBF-TF-IDF)',  'results_sklearn.csv'),
    ('2_cbf_gensim.py',     'Gensim (CBF-Doc2Vec)',       'results_gensim.csv'),
    ('3_cf_surprise.py',    'CF-NMF (sklearn)',           'results_surprise.csv'),
    ('4_hybrid_sklearn.py', 'Hybrid-sklearn (CF+CBF)',    'results_hybrid_sklearn.csv'),
    ('5_hybrid_tensorflow.py','TensorFlow (Hybrid-DL)',   'results_tensorflow.csv'),
]

TIMEOUT = 600   # secondi per ogni script
FINAL_FILE = os.path.join(OUT_DIR, 'comparison_all_models.csv')


# ========== MAIN ==========

def main():
    print("=" * 100)
    print("🚀 SCRIPT MASTER — Tutti i Modelli di Raccomandazione")
    print("=" * 100)
    print(f"\n📅 Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Script dir: {SCRIPT_DIR}")
    print(f"📁 Output dir: {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    completed   = []   # modelli terminati con successo
    failed      = []   # modelli falliti

    for script_file, model_name, result_csv in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_file)

        if not os.path.exists(script_path):
            print(f"\n❌ Script non trovato: {script_path}")
            failed.append(model_name)
            continue

        print("\n" + "=" * 100)
        print(f"▶  Esecuzione: {model_name}")
        print("=" * 100)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            elapsed = time.perf_counter() - t0

            # stampa output del sottoprocesso
            if proc.stdout:
                print(proc.stdout)
            if proc.returncode != 0:
                print(f"⚠️  Exit code {proc.returncode}")
                if proc.stderr:
                    print("STDERR:", proc.stderr[-2000:])   # ultime 2000 char
                failed.append(model_name)
            else:
                completed.append((model_name, result_csv))
                print(f"✅ Completato in {elapsed:.1f}s")

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout dopo {TIMEOUT}s")
            failed.append(model_name)
        except Exception as e:
            print(f"❌ Errore: {e}")
            failed.append(model_name)

    # ===== RACCOLTA RISULTATI =====
    print("\n" + "=" * 100)
    print("📊 RACCOLTA RISULTATI COMPARATIVI")
    print("=" * 100)

    comparison_rows = []
    for model_name, result_csv in completed:
        result_path = os.path.join(OUT_DIR, result_csv)
        if not os.path.exists(result_path):
            print(f"❌ {model_name}: file CSV non trovato ({result_path})")
            continue
        try:
            df_res = pd.read_csv(result_path)
            row = df_res.iloc[0].to_dict()
            # assicura che il campo Model sia presente
            if 'Model' not in row:
                row['Model'] = model_name
            comparison_rows.append(row)
            print(f"   ✅ {model_name}: risultati caricati")
        except Exception as e:
            print(f"   ❌ {model_name}: errore lettura CSV — {e}")

    if not comparison_rows:
        print("\n❌ Nessun risultato disponibile da mostrare.")
        if failed:
            print(f"\nModelli falliti: {', '.join(failed)}")
        return

    # ===== TABELLA COMPARATIVA =====
    comparison_df = pd.DataFrame(comparison_rows)

    col_order = [
        'Model', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10',
        'RMSE', 'MAE', 'Execution_Time_s', 'Num_Predictions'
    ]
    col_available = [c for c in col_order if c in comparison_df.columns]
    comparison_df = comparison_df[col_available]

    comparison_df.to_csv(FINAL_FILE, index=False)

    print("\n" + "=" * 100)
    print("📈 TABELLA COMPARATIVA FINALE")
    print("=" * 100)
    print("\n" + comparison_df.to_string(index=False))

    # ===== BEST MODELS =====
    print("\n" + "=" * 100)
    print("🏆 MIGLIORI MODELLI PER METRICA")
    print("=" * 100)

    for col in ['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']:
        if col in comparison_df.columns and comparison_df[col].notna().any():
            best_idx   = comparison_df[col].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_val   = comparison_df.loc[best_idx, col]
            print(f"   • Miglior {col:15s}: {best_model} ({best_val:.4f})")

    for col, direction in [('RMSE', 'min'), ('MAE', 'min'), ('Execution_Time_s', 'min')]:
        if col in comparison_df.columns and comparison_df[col].notna().any():
            best_idx   = (comparison_df[col].idxmin() if direction == 'min'
                          else comparison_df[col].idxmax())
            best_model = comparison_df.loc[best_idx, 'Model']
            best_val   = comparison_df.loc[best_idx, col]
            label = 'Miglior' if col != 'Execution_Time_s' else 'Più veloce'
            unit  = 's' if col == 'Execution_Time_s' else ''
            print(f"   • {label} {col:20s}: {best_model} ({best_val:.4f}{unit})")

    if failed:
        print(f"\n⚠️  Modelli non completati: {', '.join(failed)}")
        print("   → Verifica che le librerie siano installate (vedi README_SCRIPTS.md)")

    print(f"\n✅ Tabella comparativa salvata: {FINAL_FILE}")
    print("\n" + "=" * 100)
    print("✅ ESECUZIONE MASTER COMPLETATA")
    print("=" * 100)


if __name__ == '__main__':
    main()
