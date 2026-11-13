"""Statistiques complémentaires (VAD oracle, p=0.1) : corrélations ΔDER, comparaisons à fichiers constants."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "results/diag_k_oracle_vs_eigengap"
w = pd.read_csv(f"{OUT}/par_fichier_eig_vs_oracle_sample90.csv")
s = pd.read_csv(f"{OUT}/speakers_sample90.csv")
r = pd.read_csv(f"{OUT}/runs_sample90.csv")
for v in ["oracle", "energy", "stack_devtrain"]:
    x = w[w.vad == v].copy()
    sp = s[(s.vad == v) & (s.p_neighbors == 0.1)]
    g = sp.groupby("uri").agg(min_share=("ref_share", "min"), n_small=("n_win_major", lambda a: 0),
                              n_below_p=("speaker", "size"))
    g["n_below_p"] = sp.assign(b=sp.n_win_major < sp.p).groupby("uri").b.sum()
    g["dom_share"] = sp.groupby("uri").ref_share.max()
    x = x.merge(g, on="uri")
    u = x[x.regime == "k_eig<n_ref"]
    print(f"--- VAD {v} : fichiers k_eig<n_ref (n={len(u)}) : Spearman ΔDER vs ...")
    for c in ["gap_at_nref", "n_below_p", "dom_share", "min_share", "n_ref_speakers", "frac_win_mixed_eig"]:
        rho, pv = spearmanr(u[c], u.dDER, nan_policy="omit")
        print(f"   {c:20s} rho={rho:+.3f} p={pv:.3g}")
    print(f"   Δcoverage moyen {u.coverage_orc.mean() - u.coverage_eig.mean():+.3f}, Δpurity moyen {u.purity_orc.mean() - u.purity_eig.mean():+.3f}, "
          f"n_clusters_dominant moyen eig {u.n_clusters_dominant_eig.mean():.2f} -> oracle {u.n_clusters_dominant_orc.mean():.2f}, "
          f"n_spk_isolated moyen {u.n_spk_isolated_eig.mean():.2f} -> {u.n_spk_isolated_orc.mean():.2f} (n_ref moyen {u.n_ref_speakers.mean():.2f})")
    print(f"   fichiers k_eig<n_ref où le cluster du dominant est coupé (n_clusters_dominant_orc>=2) : "
          f"{(u.n_clusters_dominant_orc >= 2).sum()}/{len(u)} ; parmi eux ΔDER>+1pt : {((u.n_clusters_dominant_orc >= 2) & (u.dDER > .01)).sum()}")

# mêmes fichiers : eigengap / oracle / n_ref-1 / meilleur k (oracle des oracles), VAD oracle
sw = r[(r.vad == "oracle") & r.k_mode.str.startswith("k=")]
files = sorted(w[(w.vad == "oracle") & (w.n_ref_speakers >= 2)].uri)
def mic(df):
    return 100 * (df.FA_s + df.miss_s + df.conf_s).sum() / df.total_s.sum(), 100 * df.conf_s.sum() / df.total_s.sum()
o = r[(r.vad == "oracle") & (r.p_neighbors == 0.1)]
rows = {"eigengap": o[(o.k_mode == "eigengap") & o.uri.isin(files)],
        "k = n_ref": o[(o.k_mode == "oracle") & o.uri.isin(files)],
        "k = n_ref - 1": sw[sw.uri.isin(files) & (sw.k_forced == sw.n_ref_speakers - 1)],
        "k = max(1, n_ref - 2)": sw[sw.uri.isin(files) & (sw.k_forced == np.maximum(1, sw.n_ref_speakers - 2))],
        "meilleur k par fichier (oracle DER)": sw[sw.uri.isin(files)].loc[lambda d: d.groupby("uri").DER.idxmin()]}
print(f"\n--- VAD oracle, p=0.1, mêmes {len(files)} fichiers (n_ref >= 2) ---")
for k, d in rows.items():
    a, c = mic(d); print(f"   {k:38s} n={d.uri.nunique():3d}  DER {a:.2f} %  confusion {c:.2f} %")
