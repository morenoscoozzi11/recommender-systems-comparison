"""
SCRIPT 10: ANALISI DEL FALLIMENTO DI NMF E HYBRID SVD
Dimostra perché NMF e Hybrid vengono battuti dalla baseline su dataset sparsi.

OUTPUT:
  analisi_fallimento_nmf.png

PREREQUISITI:
  - amazon_cf_10k.csv, amazon_cf_40k.csv  (generato da preprocess_amazon.py)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, TruncatedSVD

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'RANDOM_STATE'         : 42,
    'TEST_SIZE'            : 0.2,
    'MIN_USER_INTERACTIONS': 5,
    'MIN_ITEM_INTERACTIONS': 5,
    'NMF_N_COMPONENTS'     : 20,
    'SVD_N_COMPONENTS'     : 20,
    'NMF_MAX_ITER'         : 300,
    'OUTPUT_PATH'          : os.path.join(BASE_DIR, 'analisi_fallimento_nmf.png'),
}

SIZES = ['10k', '40k']


def load_and_filter(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['userID', 'itemID'], keep='first')
    uc = df['userID'].value_counts()
    ic = df['itemID'].value_counts()
    df = df[df['userID'].isin(uc[uc >= CONFIG['MIN_USER_INTERACTIONS']].index)]
    df = df[df['itemID'].isin(ic[ic >= CONFIG['MIN_ITEM_INTERACTIONS']].index)]
    df = df.groupby(['userID', 'itemID'])['rating'].mean().reset_index()
    return df


def build_matrix(df):
    matrix = df.pivot_table(index='userID', columns='itemID',
                            values='rating', fill_value=0)
    return matrix.values.astype(np.float64), matrix


def main():
    print("=" * 70)
    print("📊  ANALISI FALLIMENTO NMF E HYBRID SVD")
    print("=" * 70)

    # ── raccoglie dati per ogni dimensione ───────────────────────────────────
    svd_variance   = {}   # varianza spiegata SVD per dimensione
    nmf_pred_dist  = {}   # distribuzione predizioni NMF su celle vuote
    sparsity_vals  = {}   # sparsità per dimensione
    interactions_df = {}  # distribuzione interazioni per utente

    for size in SIZES:
        path = os.path.join(BASE_DIR, f'amazon_cf_{size}.csv')
        if not os.path.exists(path):
            print(f"⚠️  File mancante: {path}")
            continue

        print(f"\n📁 Dataset {size}...")
        df = load_and_filter(path)
        R, matrix = build_matrix(df)

        sparsity = (R == 0).sum() / R.size * 100
        sparsity_vals[size] = sparsity
        print(f"   Sparsità: {sparsity:.1f}% | Shape: {R.shape}")

        # distribuzione interazioni per utente
        interactions_df[size] = matrix.apply(lambda r: (r > 0).sum(), axis=1).values

        # SVD — varianza spiegata
        n_comp = min(CONFIG['SVD_N_COMPONENTS'], min(R.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp,
                           random_state=CONFIG['RANDOM_STATE'])
        svd.fit(R)
        svd_variance[size] = svd.explained_variance_ratio_.sum() * 100
        print(f"   SVD varianza spiegata: {svd_variance[size]:.1f}%")

        # NMF — distribuzione predizioni su celle VUOTE
        train, _ = train_test_split(df, test_size=CONFIG['TEST_SIZE'],
                                    random_state=CONFIG['RANDOM_STATE'])
        R_train, _ = build_matrix(train)
        n_comp_nmf = min(CONFIG['NMF_N_COMPONENTS'], min(R_train.shape) - 1)
        nmf = NMF(n_components=n_comp_nmf, max_iter=CONFIG['NMF_MAX_ITER'],
                  random_state=CONFIG['RANDOM_STATE'], init='nndsvda')
        W = nmf.fit_transform(R_train)
        H = nmf.components_
        R_approx = W @ H
        # solo celle vuote nel training
        empty_mask = R_train == 0
        nmf_pred_dist[size] = R_approx[empty_mask]
        print(f"   NMF predizioni celle vuote — media: "
              f"{nmf_pred_dist[size].mean():.4f}, "
              f"std: {nmf_pred_dist[size].std():.4f}")

    if not svd_variance:
        print("❌  Nessun dato disponibile.")
        return

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    sizes_available = [s for s in SIZES if s in svd_variance]
    colors = ['steelblue', 'tomato']

    # 1. Varianza spiegata SVD vs dimensione dataset
    vars_ = [svd_variance[s] for s in sizes_available]
    bars  = ax1.bar(sizes_available, vars_,
                    color=colors[:len(sizes_available)],
                    edgecolor='white', linewidth=0.5, width=0.4)
    for bar, v in zip(bars, vars_):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=11,
                 fontweight='bold')
    ax1.set_xlabel('Dimensione dataset', fontsize=11)
    ax1.set_ylabel('Varianza spiegata (%)', fontsize=11)
    ax1.set_title('Varianza spiegata da SVD\nvs dimensione dataset',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100); ax1.grid(axis='y', alpha=0.3)

    # 2. Distribuzione predizioni NMF su celle vuote
    for i, size in enumerate(sizes_available):
        preds = nmf_pred_dist[size]
        sample = preds[np.random.choice(len(preds),
                                        min(5000, len(preds)), replace=False)]
        ax2.hist(sample, bins=50, alpha=0.6, color=colors[i],
                 label=f'{size} (media={preds.mean():.3f})',
                 edgecolor='none', density=True)
    ax2.set_xlabel('Score predetto NMF (celle vuote)', fontsize=11)
    ax2.set_ylabel('Densità', fontsize=11)
    ax2.set_title('Predizioni NMF sulle celle vuote\n(quasi zero = ranking casuale)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    # 3. Distribuzione interazioni per utente (scala log)
    for i, size in enumerate(sizes_available):
        if size not in interactions_df:
            continue
        inter = interactions_df[size]
        ax3.hist(inter, bins=30, alpha=0.6, color=colors[i],
                 label=f'{size} (media={inter.mean():.1f})',
                 edgecolor='none', density=True)
    ax3.set_xlabel('N° interazioni per utente', fontsize=11)
    ax3.set_ylabel('Densità', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_title('Distribuzione interazioni\nper utente (scala log)',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

    # 4. Sparsità vs varianza spiegata (scatter con annotazioni)
    for i, size in enumerate(sizes_available):
        ax4.scatter(sparsity_vals[size], svd_variance[size],
                    color=colors[i], s=120, zorder=5,
                    label=size)
        ax4.annotate(f'  {size}\n  ({sparsity_vals[size]:.1f}%, '
                     f'{svd_variance[size]:.1f}%)',
                     (sparsity_vals[size], svd_variance[size]),
                     fontsize=9)
    ax4.set_xlabel('Sparsità matrice utente-item (%)', fontsize=11)
    ax4.set_ylabel('Varianza spiegata SVD (%)', fontsize=11)
    ax4.set_title('Sparsità vs Varianza spiegata\n(più sparso = SVD meno efficace)',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    fig.suptitle('Analisi del fallimento di NMF e Hybrid SVD',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(CONFIG['OUTPUT_PATH'], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅  Grafico salvato: {CONFIG['OUTPUT_PATH']}")


if __name__ == '__main__':
    main()
