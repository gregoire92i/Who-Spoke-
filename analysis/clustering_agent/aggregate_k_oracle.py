"""Agrégats micro et tableaux par fichier à partir de runs_<tag>.csv / speakers_<tag>.csv.
Usage : python analysis/clustering_agent/aggregate_k_oracle.py [tag]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/diag_k_oracle_vs_eigengap")
tag = sys.argv[1] if len(sys.argv) > 1 else "sample90"
r = pd.read_csv(OUT / f"runs_{tag}.csv")
s = pd.read_csv(OUT / f"speakers_{tag}.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)


def micro(g):
    t = g.total_s.sum()
    return pd.Series({
        "n_files": g.uri.nunique(),
        "DER_%": 100 * (g.FA_s + g.miss_s + g.conf_s).sum() / t,
        "DER_forgiving_%": 100 * g.f_err_s.sum() / g.f_total_s.sum(),
        "JER_%": 100 * g.jer_err.sum() / g.jer_count.sum(),
        "FA_%": 100 * g.FA_s.sum() / t, "miss_%": 100 * g.miss_s.sum() / t, "conf_%": 100 * g.conf_s.sum() / t,
        "conf_impurity_%": 100 * g.conf_impurity_s.sum() / t, "conf_fragment_%": 100 * g.conf_fragment_s.sum() / t,
        "purity_%": 100 * g.pur_num.sum() / g.pur_den.sum(), "coverage_%": 100 * g.cov_num.sum() / g.cov_den.sum(),
        "ARI_pure_mean": g.ARI_pure.mean(), "AMI_pure_mean": g.AMI_pure.mean(),
        "mean_abs_k_err": (g.k_hyp - g.n_ref_speakers).abs().mean(),
        "mean_k_err": (g.k_hyp - g.n_ref_speakers).mean(),
        "spk_isolated_total": g.n_spk_isolated.sum(), "clusters_redundant_total": g.n_clusters_redundant.sum(),
    })


main = r[~r.k_mode.str.startswith("k=")]
agg = main.groupby(["vad", "p_neighbors", "k_mode"]).apply(micro, include_groups=False).reset_index()
agg.to_csv(OUT / f"agregats_micro_{tag}.csv", index=False)
print("=== agrégats micro (90 fichiers dev) ===")
print(agg.round(3).to_string(index=False))

# ---- par fichier, eigengap vs oracle (p = 0.1) ----
cols = ["k_hyp", "DER", "DER_forgiving", "JER", "conf_s", "conf_impurity_s", "conf_fragment_s", "purity",
        "coverage", "ARI_all", "ARI_pure", "AMI_pure", "n_spk_isolated", "n_clusters_dominant",
        "largest_cluster_frac", "frac_win_mixed", "total_s"]
base = ["uri", "vad", "duration_s", "n_ref_speakers", "n_windows", "p", "gap_at_keig", "gap_at_nref"]
p1 = r[r.p_neighbors == 0.1]
e = p1[p1.k_mode == "eigengap"][base + cols]
o = p1[p1.k_mode == "oracle"][["uri", "vad"] + cols]
w = e.merge(o, on=["uri", "vad"], suffixes=("_eig", "_orc"))
w["k_err_eig"] = w.k_hyp_eig - w.n_ref_speakers
w["regime"] = np.select([w.k_err_eig < 0, w.k_err_eig > 0], ["k_eig<n_ref", "k_eig>n_ref"], "k_eig=n_ref")
w["dDER"] = w.DER_orc - w.DER_eig
w["dconf_s"] = w.conf_s_orc - w.conf_s_eig
w.to_csv(OUT / f"par_fichier_eig_vs_oracle_{tag}.csv", index=False)

for v in ["energy", "oracle", "stack_devtrain"]:
    x = w[w.vad == v]
    print(f"\n=== VAD {v} : Δ (k oracle − eigengap), p=0.1 ===")
    g = x.groupby("regime").agg(n=("uri", "size"), dconf_s=("dconf_s", "sum"),
                                n_worse=("dDER", lambda d: (d > 0.01).sum()),
                                n_better=("dDER", lambda d: (d < -0.01).sum()),
                                median_dDER=("dDER", "median"))
    g["dconf_pts_of_total"] = 100 * g.dconf_s / x.total_s_eig.sum()
    print(g.round(3).to_string())
    print(f"Δconf total = {100 * x.dconf_s.sum() / x.total_s_eig.sum():+.2f} pts ; "
          f"top-10 fichiers = {100 * x.dconf_s.nlargest(10).sum() / x.total_s_eig.sum():+.2f} pts ; "
          f"fichiers avec ΔDER>+1pt : {(x.dDER > 0.01).sum()}, <−1pt : {(x.dDER < -0.01).sum()}, |Δ|≤1pt : {(x.dDER.abs() <= 0.01).sum()}")
    print("k_eig − n_ref :", x.k_err_eig.value_counts().sort_index().to_dict())

# ---- locuteurs : qui gagne / perd avec le k oracle (VAD oracle, p=0.1) ----
for v in ["oracle", "energy", "stack_devtrain"]:
    sp = s[(s.vad == v) & (s.p_neighbors == 0.1)].merge(w[w.vad == v][["uri", "regime"]], on="uri")
    sp["win_over_p"] = sp.n_win_major / sp.p
    sp["bin"] = pd.cut(sp.win_over_p, [-0.01, 0, 0.5, 1, 2, 5, np.inf],
                       labels=["0", "]0,0.5]", "]0.5,1]", "]1,2]", "]2,5]", ">5"])
    print(f"\n=== VAD {v} : locuteurs ({len(sp)}), par n_win_major / p ===")
    print(sp.groupby("bin", observed=False).agg(n=("speaker", "size"), ref_time_s=("ref_time_s", "sum"),
          neighbor_purity=("neighbor_purity", "mean"), isolated_eig=("isolated_eig", "mean"),
          isolated_oracle=("isolated_oracle", "mean"), recall_eig=("recall_eig", "mean"),
          recall_oracle=("recall_oracle", "mean")).round(3).to_string())
    print("dominant vs autres (recall moyen, pondéré par temps) :")
    for d, g in sp.groupby("is_dominant"):
        print(f"  dominant={d}: n={len(g)}, recall_eig={np.average(g.recall_eig, weights=g.ref_time_s):.3f}, "
              f"recall_oracle={np.average(g.recall_oracle, weights=g.ref_time_s):.3f}, "
              f"temps={g.ref_time_s.sum():.0f} s")
    print(f"  locuteurs sans aucune window majoritaire : {(sp.n_win_major == 0).sum()} / {len(sp)} "
          f"({sp[sp.n_win_major == 0].ref_time_s.sum():.0f} s de parole de réf.) ; "
          f"avec n_win_major < p : {(sp.n_win_major < sp.p).sum()}")

# ---- balayage de k (VAD oracle) ----
sw = r[r.k_mode.str.startswith("k=")].copy()
sw["k_minus_nref"] = sw.k_forced - sw.n_ref_speakers
sw.to_csv(OUT / f"sweep_k_{tag}.csv", index=False)
multi = sw[sw.n_ref_speakers >= 2]
print("\n=== balayage k (VAD oracle, p=0.1, fichiers n_ref>=2) : micro par k − n_ref ===")
print(multi.groupby("k_minus_nref").apply(micro, include_groups=False)[
    ["n_files", "DER_%", "conf_%", "purity_%", "coverage_%", "ARI_pure_mean"]].round(2).to_string())
best = sw.loc[sw.groupby("uri").DER.idxmin()][["uri", "n_ref_speakers", "k_forced"]]
best["best_k_minus_nref"] = best.k_forced - best.n_ref_speakers
print("k qui minimise le DER (par fichier) − n_ref :", best.best_k_minus_nref.value_counts().sort_index().to_dict())
best.to_csv(OUT / f"best_k_par_fichier_{tag}.csv", index=False)
