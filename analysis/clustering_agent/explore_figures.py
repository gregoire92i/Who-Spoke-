"""Figures exploratoires (neutres) : k oracle vs eigengap, une sous-section par piste.
Sorties : results/diag_k_oracle_vs_eigengap/figures_exploration/pXX_*/
Entrées : CSV de diag_k_oracle.py, aggregate_k_oracle.py, explore_compute.py (tag sample90).
Usage : python analysis/clustering_agent/explore_figures.py [--cases ...] [--skip-cases]
Seeds : KMeans 42 (sauf p10), PCA/t-SNE random_state=0. Backend Agg."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_k_oracle as dk
from audio_features import load_audio, FFT, compute_mfcc
from embeddings import compute_embeddings, timeline_to_mask
from post_treatment import windows_to_annotation
from vad import best_vad

OUT = dk.OUT
FIG = OUT / "figures_exploration"
TAG = "sample90"
DPI = 150
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000",
       "#999933", "#882255", "#44AA99", "#117733", "#332288", "#AA4499", "#88CCEE", "#DDCC77",
       "#661100", "#6699CC", "#AA4466", "#4477AA"]
GREY = "#BBBBBB"
REG_COL = {"k_eig<n_ref": "#D55E00", "k_eig=n_ref": "#777777", "k_eig>n_ref": "#0072B2"}
# fichiers représentatifs : pires (oracle dégrade), oracle aide (k sous-estimé), eigengap surestime, k correct
CASES = {"kdfqk": "oracle dégrade (20 réf.)", "bauzd": "oracle dégrade", "uvnmy": "oracle dégrade (locuteurs de 2 s)",
         "evtyi": "oracle dégrade (2 réf.)", "hkzpa": "oracle aide (eig=1)", "hgdez": "oracle aide (eig=1, 10 réf.)",
         "oxxwk": "eigengap surestime", "zfkap": "eigengap surestime", "odkzj": "k correct (7 réf.)",
         "qygfk": "k correct (4 réf.)"}


def sec(name):
    d = FIG / name; d.mkdir(parents=True, exist_ok=True); return d


def save(fig, d, name):
    fig.tight_layout(); fig.savefig(d / name, dpi=DPI); plt.close(fig)


def micro_conf(g):
    return 100 * g.conf_s.sum() / g.total_s.sum()


def micro_der(g):
    return 100 * (g.FA_s + g.miss_s + g.conf_s).sum() / g.total_s.sum()


def regime_legend(ax, x=None):
    ax.legend(handles=[Patch(color=c, label=g + (f" (n={(x.regime == g).sum()})" if x is not None else ""))
                       for g, c in REG_COL.items()], fontsize=7)


# ============================================================ chargement
def load():
    W = pd.read_csv(OUT / f"par_fichier_eig_vs_oracle_{TAG}.csv")
    S = pd.read_csv(OUT / f"speakers_{TAG}.csv")
    SW = pd.read_csv(OUT / f"sweep_k_{TAG}.csv")
    EQ = pd.read_csv(OUT / f"embeddings_quality_{TAG}.csv")
    BK = pd.read_csv(OUT / f"best_k_par_fichier_{TAG}.csv")
    RU = pd.read_csv(OUT / f"runs_{TAG}.csv")
    E = pd.read_csv(OUT / f"explore_runs_{TAG}.csv")
    F = pd.read_csv(OUT / f"explore_files_{TAG}.csv")
    sp = S[S.p_neighbors == 0.1]
    nr = sp.assign(r=sp.n_win_major < sp.p).groupby(["uri", "vad"]).r.sum().rename("n_rare").reset_index()
    W = W.merge(nr, on=["uri", "vad"]).merge(F[["uri", "speech_s", "overlap_rate", "dom_share", "mean_turn_s", "n_turns"]], on="uri")
    eig_info = RU[(RU.k_mode == "eigengap") & (RU.p_neighbors == 0.1)][["uri", "vad", "lambda_nref", "n_eig_below_0.05"]]
    W = W.merge(eig_info, on=["uri", "vad"])
    return W, S, SW, EQ, BK, E, F


# ============================================================ p01 répartition
def p01(W):
    d = sec("p01_repartition")
    for v in ["oracle", "energy", "stack_devtrain"]:
        x = W[W.vad == v].sort_values("dconf_s", ascending=False).reset_index(drop=True)
        tot = x.total_s_eig.sum()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(x)), 100 * x.dconf_s / tot, color=[REG_COL[g] for g in x.regime])
        ax.set_ylabel("Δ confusion du fichier (pts de DER micro)")
        ax2 = ax.twinx()
        ax2.plot(range(len(x)), 100 * x.dconf_s.cumsum() / tot, "k-", lw=1.2, label="cumul")
        ax2.set_ylabel("cumul (pts de DER micro)")
        ax.set_xticks(range(len(x)), x.uri, rotation=90, fontsize=6)
        ax.axhline(0, color="k", lw=.5); regime_legend(ax, x)
        ax.set_title(f"VAD {v} : contribution de chaque fichier à Δconfusion micro (k oracle − eigengap), "
                     f"trié décroissant ; total {100 * x.dconf_s.sum() / tot:+.2f} pts")
        save(fig, d, f"delta_conf_trie_cumule_vad_{v}.png")
    x = W[W.vad == "oracle"].copy()
    x["duration_min"] = x.duration_s / 60
    covs = [("n_ref_speakers", "n_ref"), ("k_err_eig", "k_eig − n_ref"), ("duration_min", "durée (min)"),
            ("dom_share", "part du locuteur dominant"), ("n_rare", "nb locuteurs avec < p windows"),
            ("overlap_rate", "taux d'overlap (part de la parole)"), ("speech_s", "durée de parole réf. (s)"),
            ("mean_turn_s", "durée moyenne d'un tour (s)"), ("n_windows_eig", "n windows (p = max(5, 10 %))")]
    if "n_windows_eig" not in x:
        x["n_windows_eig"] = x["n_windows"]
    for yname, ylab in [("dDER", "ΔDER moderne (pts)"), ("dconf_s", "Δ confusion (s)")]:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for ax, (c, lab) in zip(axes.flat, covs):
            yy = 100 * x[yname] if yname == "dDER" else x[yname]
            ax.scatter(x[c], yy, c=[REG_COL[g] for g in x.regime], s=18, alpha=.8)
            for u in CASES:
                r = x[x.uri == u]
                if len(r): ax.annotate(u, (r[c].iloc[0], (100 * r[yname] if yname == "dDER" else r[yname]).iloc[0]), fontsize=6)
            rho, pv = spearmanr(x[c], x[yname])
            ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel(ylab)
            ax.set_title(f"Spearman ρ={rho:+.2f} (p={pv:.2g}), 90 fichiers", fontsize=8)
        regime_legend(axes.flat[0], x)
        fig.suptitle(f"VAD oracle, p=0.1, seed 42 : {ylab} par fichier vs caractéristiques du fichier "
                     f"(couleur = erreur de l'eigengap)", fontsize=10)
        save(fig, d, f"{yname}_vs_caracteristiques_vad_oracle.png")
    # même chose restreint au régime k_eig<n_ref (ρ intra-régime)
    u = x[x.regime == "k_eig<n_ref"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (c, lab) in zip(axes.flat, covs):
        ax.scatter(u[c], 100 * u.dDER, color=REG_COL["k_eig<n_ref"], s=18)
        rho, pv = spearmanr(u[c], u.dDER)
        ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("ΔDER (pts)")
        ax.set_title(f"ρ={rho:+.2f} (p={pv:.2g}), n={len(u)}", fontsize=8)
    fig.suptitle("Idem, seulement les fichiers où l'eigengap sous-estime k", fontsize=10)
    save(fig, d, "dDER_vs_caracteristiques_regime_sous_estime.png")


# ============================================================ p02 k
def p02(W, SW, BK):
    d = sec("p02_k")
    x = W[W.vad == "oracle"].merge(BK[["uri", "k_forced"]].rename(columns={"k_forced": "best_k"}), on="uri")
    mx = int(max(x.n_ref_speakers.max(), x.k_hyp_eig.max(), x.best_k.max()))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, col, name in [(axes[0], "k_hyp_eig", "k eigengap"), (axes[1], "best_k", "meilleur k (min DER, k ≤ n_ref+3)")]:
        H = np.zeros((mx + 1, mx + 1), int)
        for a, b in zip(x.n_ref_speakers, x[col]): H[int(b), int(a)] += 1
        ax.imshow(H, origin="lower", cmap="Blues")
        for i in range(mx + 1):
            for j in range(mx + 1):
                if H[i, j]: ax.text(j, i, H[i, j], ha="center", va="center", fontsize=6)
        ax.plot([0, mx], [0, mx], "r--", lw=.6); ax.set_xlabel("n_ref"); ax.set_ylabel(name)
        ax.set_title(f"nombre de fichiers : {name} vs n_ref", fontsize=9)
    ax = axes[2]
    j = np.random.default_rng(0).uniform(-.15, .15, len(x))
    ax.scatter(x.k_err_eig + j, x.best_k - x.n_ref_speakers - j, c=100 * x.dDER, cmap="coolwarm", vmin=-50, vmax=50, s=20)
    ax.plot([-20, 5], [-20, 5], "k--", lw=.5); ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_xlabel("k_eig − n_ref"); ax.set_ylabel("meilleur k − n_ref")
    ax.set_title("couleur = ΔDER (pts, k oracle − eigengap) ; jitter ±0,15", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-50, 50)); fig.colorbar(sm, ax=ax)
    fig.suptitle("VAD oracle, p=0.1, seed 42, 90 fichiers dev", fontsize=10)
    save(fig, d, "k_eig_best_k_vs_n_ref.png")
    # courbes par fichier représentatif
    metrics = [("DER", "DER"), ("conf_rate", "confusion / parole"), ("JER", "JER"), ("purity", "purity"), ("coverage", "coverage")]
    sw = SW.copy(); sw["conf_rate"] = sw.conf_s / sw.total_s
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for ax, (u, why) in zip(axes.flat, CASES.items()):
        s = sw[sw.uri == u].sort_values("k_forced")
        for (c, lab), col in zip(metrics, ["k", "#D55E00", "#CC79A7", "#009E73", "#0072B2"]):
            ax.plot(s.k_forced, s[c], "o-", ms=3, color=col, label=lab)
        r = x[x.uri == u].iloc[0]
        ax.axvline(r.k_hyp_eig, color="#777777", ls="--", label=f"k_eig={r.k_hyp_eig}")
        ax.axvline(r.n_ref_speakers, color="#009E73", ls=":", lw=2, label=f"n_ref={r.n_ref_speakers}")
        ax.set_title(f"{u} — {why}", fontsize=8); ax.set_xlabel("k forcé"); ax.set_ylim(0, 1.2)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Métriques en fonction de k forcé (VAD oracle, p=0.1, seed 42 ; k ≤ min(n_ref+3, 20))", fontsize=10)
    save(fig, d, "courbes_metriques_vs_k_fichiers_representatifs.png")
    # tous les fichiers : DER(k) − DER(n_ref) vs k − n_ref
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, c in zip(axes, ["DER", "coverage"]):
        for u, s in sw.groupby("uri"):
            ref = s[s.k_forced == s.n_ref_speakers][c]
            if not len(ref): continue
            reg = x[x.uri == u].regime.iloc[0]
            ax.plot(s.k_forced - s.n_ref_speakers, s[c] - ref.iloc[0], color=REG_COL[reg], alpha=.35, lw=.8)
        ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.set_xlim(-10, 3)
        ax.set_xlabel("k forcé − n_ref"); ax.set_ylabel(f"{c}(k) − {c}(n_ref)")
        ax.set_title(f"{c} relatif à k = n_ref, un trait par fichier (couleur = régime eigengap)", fontsize=9)
    regime_legend(axes[0], x)
    save(fig, d, "courbes_relatives_tous_fichiers.png")


# ============================================================ p05 / p06 / p07 / p08 / p09 / p10 / p11 globaux
def p05_global(W, EQ):
    d = sec("p05_embeddings")
    h = np.load(OUT / f"cos_hist_{TAG}.npz"); c = .5 * (h["bins"][1:] + h["bins"][:-1])
    x = W[W.vad == "oracle"].merge(EQ[EQ.vad == "oracle"], on="uri", suffixes=("", "_eq"))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    ax.plot(c, h["intra"] / h["intra"].sum(), color="#009E73", label="intra"); ax.plot(c, h["inter"] / h["inter"].sum(), color="#D55E00", label="inter")
    ax.set_title("cosinus intra / inter, windows pures, 90 fichiers poolés", fontsize=9); ax.legend(); ax.set_xlabel("cosinus")
    for ax, col, lab in [(axes[0, 1], "auc_same_vs_diff", "AUC même/différent"), (axes[0, 2], "silhouette_ref", "silhouette (labels réf.)"),
                         (axes[1, 0], "cos_intra_mean", "cos intra moyen"), (axes[1, 1], "cos_inter_mean", "cos inter moyen")]:
        ok = x[col].notna()
        ax.scatter(x[col], x.dconf_s, c=[REG_COL[g] for g in x.regime], s=18)
        rho, pv = spearmanr(x[col][ok], x.dconf_s[ok])
        ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("Δ confusion (s)")
        ax.set_title(f"{lab} vs Δconf : ρ={rho:+.2f} (p={pv:.2g}), n={ok.sum()}", fontsize=9)
    ax = axes[1, 2]
    ax.scatter(x.silhouette_eigengap, x.silhouette_oracle, c=[REG_COL[g] for g in x.regime], s=18)
    ax.plot([-.5, .8], [-.5, .8], "k--", lw=.5); ax.set_xlabel("silhouette labels eigengap"); ax.set_ylabel("silhouette labels k oracle")
    ax.set_title("Silhouette des partitions prédites (windows pures)", fontsize=9)
    regime_legend(axes[0, 1], x)
    fig.suptitle("VAD oracle : qualité intrinsèque des embeddings (centrés-réduits par fichier puis L2) vs Δconfusion", fontsize=10)
    save(fig, d, "global_qualite_embeddings_vs_delta.png")
    # Fisher par dimension
    fc = [f"fisher_d{i:02d}" for i in range(26)]
    fr = EQ[EQ.vad == "oracle"][fc].dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot([fr[c] for c in fc], showfliers=False)
    ax.set_xticks(range(1, 27), [f"μc{i}" for i in range(13)] + [f"σc{i}" for i in range(13)], rotation=60, fontsize=7)
    ax.set_ylabel("ratio de Fisher (inter / intra)"); ax.set_title(f"Ratio de Fisher par dimension de l'embedding (moyennes puis écarts-types des 13 MFCC), {len(fr)} fichiers", fontsize=9)
    save(fig, d, "global_fisher_par_dimension.png")


def p06_global(W):
    d = sec("p06_affinite_spectre")
    x = W[W.vad == "oracle"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, col, lab in [(axes[0], "gap_at_nref", "λ(n_ref+1) − λ(n_ref)"), (axes[1], "lambda_nref", "λ(n_ref)"),
                         (axes[2], "n_eig_below_0.05", "nb valeurs propres < 0,05"), (axes[3], "gap_at_keig", "écart propre retenu par l'eigengap")]:
        ok = x[col].notna()
        ax.scatter(x[col], 100 * x.dDER, c=[REG_COL[g] for g in x.regime], s=18)
        rho, pv = spearmanr(x[col][ok], x.dDER[ok])
        ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("ΔDER (pts)")
        ax.set_title(f"ρ={rho:+.2f} (p={pv:.2g})", fontsize=9)
    regime_legend(axes[0], x)
    fig.suptitle("VAD oracle, p=0.1 : indicateurs du spectre du laplacien vs ΔDER", fontsize=10)
    save(fig, d, "global_spectre_vs_delta.png")


def p07_global(S, E):
    d = sec("p07_locuteurs_rares")
    sp = S[(S.vad == "oracle") & (S.p_neighbors == 0.1) & (S.n_ref_speakers >= 2)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for flag, col, lab in [((~sp.isolated_eig) & (~sp.isolated_oracle), GREY, "isolé ni eig ni oracle"),
                           ((~sp.isolated_eig) & sp.isolated_oracle, "#009E73", "isolé seulement avec k oracle"),
                           (sp.isolated_eig & ~sp.isolated_oracle, "#D55E00", "isolé seulement avec eigengap"),
                           (sp.isolated_eig & sp.isolated_oracle, "#0072B2", "isolé dans les deux")]:
        ax.scatter((sp.n_win_major[flag] + .5) / sp.p[flag], sp.ref_time_s[flag], s=12, color=col, label=f"{lab} ({flag.sum()})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1, color="k", ls="--", lw=.6)
    ax.set_xlabel("(n windows majoritaires + 0,5) / p"); ax.set_ylabel("temps de parole réf. (s)"); ax.legend(fontsize=7)
    ax.set_title("Chaque point = un locuteur ; « isolé » = majoritaire d'au moins un cluster", fontsize=9)
    ax = axes[0, 1]
    ax.scatter(sp.recall_eig, sp.recall_oracle, s=8 + 0.05 * sp.ref_time_s, c=np.where(sp.is_dominant, "#D55E00", "#0072B2"), alpha=.6)
    ax.plot([0, 1], [0, 1], "k--", lw=.5); ax.set_xlabel("rappel eigengap"); ax.set_ylabel("rappel k oracle")
    ax.set_title("Rappel par locuteur sous l'appariement du DER (orange = dominant ; taille ∝ temps)", fontsize=9)
    ax = axes[1, 0]
    e = E[(E.vad == "oracle") & (E.p_neighbors == 0.1) & (E.seed == 42)]
    for m, col in [("eigengap", "#777777"), ("oracle", "#009E73")]:
        ax.hist(e[e.k_mode == m].time_ref_unmapped_s, bins=np.r_[0, np.logspace(0, 3.5, 25)], alpha=.6, color=col, label=f"{m} : total {e[e.k_mode == m].time_ref_unmapped_s.sum():.0f} s")
    ax.set_xscale("symlog"); ax.set_xlabel("temps (s) des locuteurs réf. non appariés, par fichier"); ax.set_ylabel("fichiers"); ax.legend()
    ax.set_title("Parole des locuteurs « perdus » (aucun cluster apparié)", fontsize=9)
    ax = axes[1, 1]
    for m, col in [("eigengap", "#777777"), ("oracle", "#009E73")]:
        v = e[e.k_mode == m].n_ref_unmapped.value_counts().sort_index()
        ax.bar(v.index + (-.2 if m == "eigengap" else .2), v.values, .4, color=col, label=m)
    ax.set_xlabel("nb de locuteurs réf. non appariés dans le fichier"); ax.set_ylabel("fichiers"); ax.legend()
    fig.suptitle("VAD oracle, p=0.1, seed 42 : locuteurs rares", fontsize=10)
    save(fig, d, "global_locuteurs_rares.png")
    # windows par locuteur selon la VAD
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, col in [("oracle", "#0072B2"), ("energy", "#E69F00"), ("stack_devtrain", "#009E73")]:
        s = S[(S.vad == v) & (S.p_neighbors == 0.1)]
        ax.ecdf(s.n_win_major / s.p, label=f"VAD {v} (n<p : {(s.n_win_major < s.p).sum()}, 0 window : {(s.n_win_major == 0).sum()})", color=col)
    ax.set_xscale("symlog", linthresh=.1); ax.axvline(1, color="k", ls="--", lw=.6)
    ax.set_xlabel("n windows majoritaires / p"); ax.set_ylabel("fonction de répartition (379 locuteurs)"); ax.legend(fontsize=7)
    save(fig, d, "global_windows_par_locuteur_selon_vad.png")


def p08_global(E, W):
    d = sec("p08_overlap_frontieres")
    e = E[(E.p_neighbors == 0.1) & (E.seed == 42) & E.k_mode.isin(["eigengap", "oracle"])].merge(W[W.vad == "oracle"][["uri", "regime"]], on="uri")
    for v in ["oracle", "stack_devtrain"]:
        x = e[e.vad == v]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cats = ["overlap"] + [f"bnd_{b}" for b in ["0-0.5s", "0.5-1s", "1-2s", ">2s"]]
        ax = axes[0]
        for i, m in enumerate(["eigengap", "oracle"]):
            g = x[x.k_mode == m]
            vals = [g[f"cf_{c}_s"].sum() for c in cats]
            ax.bar(np.arange(len(cats)) + (i - .5) * .4, vals, .4, label=m, color=["#777777", "#009E73"][i])
        ax.set_xticks(range(len(cats)), ["overlap", "frontière 0-0,5 s", "0,5-1 s", "1-2 s", "> 2 s"], fontsize=8)
        ax.set_ylabel("confusion (s), 90 fichiers"); ax.legend(); ax.set_title("Où est la confusion ? (secondes)", fontsize=9)
        ax = axes[1]
        for i, m in enumerate(["eigengap", "oracle"]):
            g = x[x.k_mode == m]
            rates = [g[f"cf_{c}_s"].sum() / g[f"speech_{c}_s"].sum() for c in cats]
            ax.bar(np.arange(len(cats)) + (i - .5) * .4, rates, .4, label=m, color=["#777777", "#009E73"][i])
        ax.set_xticks(range(len(cats)), ["overlap", "0-0,5 s", "0,5-1 s", "1-2 s", "> 2 s"], fontsize=8)
        ax.set_ylabel("taux de confusion dans la zone"); ax.set_title("Taux = confusion / parole de la zone", fontsize=9)
        ax = axes[2]
        for i, m in enumerate(["eigengap", "oracle"]):
            g = x[x.k_mode == m]
            r = [g.cf_disputed_s.sum() / g.speech_disputed_s.sum(), g.cf_undisputed_s.sum() / (g.speech_labeled_s.sum() - g.speech_disputed_s.sum())]
            ax.bar(np.arange(2) + (i - .5) * .4, r, .4, color=["#777777", "#009E73"][i], label=f"{m} : disputées {100 * g.speech_disputed_s.sum() / g.speech_labeled_s.sum():.1f} % de la parole")
        ax.set_xticks([0, 1], ["trames disputées\n(windows de labels ≠)", "non disputées"]); ax.set_ylabel("taux de confusion"); ax.legend(fontsize=7)
        fig.suptitle(f"VAD {v}, p=0.1, seed 42 : confusion selon l'overlap, la distance à une frontière de tour, et le vote des windows "
                     f"(décompte trame par trame, ≈ confusion pyannote)", fontsize=10)
        save(fig, d, f"global_overlap_frontieres_vad_{v}.png")
    # par régime et par fichier
    x = e[e.vad == "oracle"].pivot_table(index=["uri", "regime"], columns="k_mode",
                                          values=["cf_overlap_s", "cf_bnd_0-0.5s_s", "cf_bnd_>2s_s", "cf_disputed_s", "cf_frames_s"]).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    dtot = x[("cf_frames_s", "oracle")] - x[("cf_frames_s", "eigengap")]
    for ax, c, lab in zip(axes, ["cf_overlap_s", "cf_bnd_0-0.5s_s", "cf_bnd_>2s_s", "cf_disputed_s"],
                          ["en overlap", "à < 0,5 s d'une frontière", "à > 2 s de toute frontière", "trames disputées"]):
        dd = x[(c, "oracle")] - x[(c, "eigengap")]
        ax.scatter(dtot, dd, c=[REG_COL[g] for g in x.regime], s=16)
        ax.plot([-250, 320], [-250, 320], "k--", lw=.5); ax.axhline(0, color="k", lw=.4)
        ax.set_xlabel("Δ confusion totale (s)"); ax.set_ylabel(f"Δ confusion {lab} (s)")
        ax.set_title(f"part du Δ total {lab} : {100 * dd.sum() / dtot.sum():.1f} %", fontsize=9)
    regime_legend(axes[0])
    fig.suptitle("VAD oracle : par fichier, quelle part du Δconfusion se situe dans chaque zone ? (diagonale = tout le Δ dans la zone)", fontsize=10)
    save(fig, d, "par_fichier_delta_par_zone.png")


def p09_global(W, E):
    d = sec("p09_vad")
    e = E[(E.p_neighbors == 0.1) & (E.seed == 42)]
    vads = ["oracle", "energy", "stack_devtrain"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for i, m in enumerate(["eigengap", "oracle"]):
        for j, (met, f) in enumerate([("DER", micro_der), ("confusion", micro_conf)]):
            vals = [f(e[(e.vad == v) & (e.k_mode == m)]) for v in vads]
            ax.bar(np.arange(3) + (2 * j + i - 1.5) * .2, vals, .2, color=["#777777", "#009E73"][i], hatch=["", "//"][j],
                   label=f"{met} {m}")
    ax.set_xticks(range(3), vads); ax.set_ylabel("% (micro)"); ax.legend(fontsize=7); ax.set_title("DER et confusion micro par VAD", fontsize=9)
    for ax, v in zip(axes[1:], ["energy", "stack_devtrain"]):
        a = W[W.vad == "oracle"].set_index("uri"); b = W[W.vad == v].set_index("uri")
        ax.scatter(a.dconf_s, b.loc[a.index].dconf_s, c=[REG_COL[g] for g in a.regime], s=16)
        ax.plot([-250, 320], [-250, 320], "k--", lw=.5)
        ax.set_xlabel("Δconfusion (s), VAD oracle"); ax.set_ylabel(f"Δconfusion (s), VAD {v}")
        ax.set_title(f"par fichier : VAD oracle vs {v}", fontsize=9)
    regime_legend(axes[1])
    fig.suptitle("La VAD change-t-elle l'effet du k oracle ? (p=0.1, seed 42 ; best_vad en train='devtrain')", fontsize=10)
    save(fig, d, "global_vad.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    types = ["cf_impurity_s", "cf_mismapped_s", "cf_unmapped_s"]; cols = ["#56B4E9", "#E69F00", "#D55E00"]
    labs = []
    for i, v in enumerate(vads):
        for j, m in enumerate(["eigengap", "oracle"]):
            g = e[(e.vad == v) & (e.k_mode == m)]; tot = g.total_s.sum(); bottom = 0
            for t, c in zip(types, cols):
                val = 100 * g[t].sum() / tot
                ax.bar(2.5 * i + j, val, .8, bottom=bottom, color=c); bottom += val
            labs.append((2.5 * i + j, f"{v}\n{m}"))
    ax.set_xticks([a for a, _ in labs], [b for _, b in labs], fontsize=7); ax.set_ylabel("confusion (pts de DER micro)")
    ax.legend(handles=[Patch(color=c, label=t[3:-2]) for t, c in zip(types, cols)])
    ax.set_title("Type de confusion par VAD (impurity : cluster apparié à son majoritaire ; mismapped : apparié à un autre ; unmapped : non apparié)", fontsize=8)
    save(fig, d, "global_types_confusion_par_vad.png")


def p10_global(E, W):
    d = sec("p10_sensibilite")
    o = E[(E.vad == "oracle") & E.k_mode.isin(["eigengap", "oracle"])]
    s = o[o.p_neighbors == 0.1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    seeds = sorted(s.seed.unique())
    for i, m in enumerate(["eigengap", "oracle"]):
        ax.plot(seeds, [micro_der(s[(s.seed == sd) & (s.k_mode == m)]) for sd in seeds], "o-", color=["#777777", "#009E73"][i], label=f"DER {m}")
        ax.plot(seeds, [micro_conf(s[(s.seed == sd) & (s.k_mode == m)]) for sd in seeds], "s--", color=["#777777", "#009E73"][i], label=f"confusion {m}")
    ax.set_xticks(seeds); ax.set_xlabel("seed KMeans"); ax.set_ylabel("% micro"); ax.legend(fontsize=7)
    ax.set_title("Seeds KMeans (VAD oracle, p=0.1)", fontsize=9)
    ax = axes[1]
    pv = s.pivot_table(index=["uri", "seed"], columns="k_mode", values="DER").reset_index()
    pv["d"] = 100 * (pv.oracle - pv.eigengap)
    g = pv.groupby("uri").d.agg(["min", "max", "median"]).sort_values("median")
    ax.vlines(range(len(g)), g["min"], g["max"], color="k", lw=1); ax.scatter(range(len(g)), g["median"], s=8, color="#D55E00")
    ax.axhline(0, color="k", lw=.4); ax.set_xticks(range(len(g)), g.index, rotation=90, fontsize=5)
    ax.set_ylabel("ΔDER (pts) : médiane et min-max sur 5 seeds"); ax.set_title("Stabilité par fichier", fontsize=9)
    ax = axes[2]
    ps = sorted(o.p_neighbors.unique())
    q = o[o.seed == 42]
    for i, m in enumerate(["eigengap", "oracle"]):
        ax.plot(ps, [micro_der(q[(q.p_neighbors == p) & (q.k_mode == m)]) for p in ps], "o-", color=["#777777", "#009E73"][i], label=f"DER {m}")
        ax.plot(ps, [micro_conf(q[(q.p_neighbors == p) & (q.k_mode == m)]) for p in ps], "s--", color=["#777777", "#009E73"][i], label=f"confusion {m}")
    ax.set_xlabel("p_neighbors"); ax.set_ylabel("% micro"); ax.legend(fontsize=7); ax.set_title("Balayage p_neighbors (seed 42)", fontsize=9)
    fig.suptitle("Sensibilité de l'écart k oracle − eigengap (VAD oracle, 90 fichiers)", fontsize=10)
    save(fig, d, "global_seeds_p_neighbors.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    for p in ps:
        pp = q[q.p_neighbors == p].pivot_table(index="uri", columns="k_mode", values="DER")
        data.append(100 * (pp.oracle - pp.eigengap).values)
    ax.boxplot(data, showfliers=False)
    for i, dd in enumerate(data):
        ax.scatter(i + 1 + np.random.default_rng(i).uniform(-.2, .2, len(dd)), dd, s=5, alpha=.4)
    ax.set_xticks(range(1, len(ps) + 1), [str(p) for p in ps]); ax.axhline(0, color="k", lw=.4)
    ax.set_xlabel("p_neighbors"); ax.set_ylabel("ΔDER par fichier (pts)"); ax.set_title("ΔDER par fichier selon p_neighbors (VAD oracle, seed 42)", fontsize=9)
    save(fig, d, "par_fichier_delta_selon_p.png")


def p11_global(E, W):
    d = sec("p11_metrique")
    e = E[(E.vad == "oracle") & (E.p_neighbors == 0.1) & (E.seed == 42)]
    modes = ["eigengap", "oracle", "eigengap_merged", "oracle_merged"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for j, (met, f) in enumerate([("DER", micro_der), ("DER forgiving", lambda g: 100 * g.f_err_s.sum() / g.f_total_s.sum()),
                                  ("JER", lambda g: 100 * g.jer_err.sum() / g.jer_count.sum()), ("confusion", micro_conf)]):
        ax.bar(np.arange(4) + (j - 1.5) * .2, [f(e[e.k_mode == m]) for m in modes], .2, label=met)
    ax.set_xticks(range(4), ["eigengap", "k oracle", "eigengap\nregroupé*", "k oracle\nregroupé*"]); ax.set_ylabel("% micro"); ax.legend(fontsize=7)
    ax.set_title("* regroupé = chaque cluster fusionné sur son locuteur majoritaire (contrôle oracle)", fontsize=8)
    ax = axes[1]
    types = ["cf_impurity_s", "cf_mismapped_s", "cf_unmapped_s"]; cols = ["#56B4E9", "#E69F00", "#D55E00"]
    ww = W[W.vad == "oracle"][["uri", "regime"]]
    ee = e.merge(ww, on="uri")
    pos = 0; ticks = []
    for reg in REG_COL:
        for m in ["eigengap", "oracle"]:
            g = ee[(ee.regime == reg) & (ee.k_mode == m)]; tot = e[e.k_mode == m].total_s.sum(); b = 0
            for t, c in zip(types, cols):
                v = 100 * g[t].sum() / tot; ax.bar(pos, v, .8, bottom=b, color=c); b += v
            ticks.append((pos, f"{reg}\n{m}")); pos += 1
        pos += .5
    ax.set_xticks([a for a, _ in ticks], [b for _, b in ticks], fontsize=6)
    ax.set_ylabel("confusion (pts de DER micro, 90 fichiers)")
    ax.legend(handles=[Patch(color=c, label=t[3:-2]) for t, c in zip(types, cols)], fontsize=7)
    ax.set_title("Type de confusion par régime de l'eigengap", fontsize=9)
    ax = axes[2]
    pv = e[e.k_mode.isin(["eigengap", "oracle"])].pivot_table(index="uri", columns="k_mode", values=["DER", "JER"])
    dD = 100 * (pv[("DER", "oracle")] - pv[("DER", "eigengap")]); dJ = 100 * (pv[("JER", "oracle")] - pv[("JER", "eigengap")])
    reg = ww.set_index("uri").loc[pv.index].regime
    ax.scatter(dD, dJ, c=[REG_COL[g] for g in reg], s=16); ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_xlabel("ΔDER (pts)"); ax.set_ylabel("ΔJER (pts)"); ax.set_title("Par fichier : DER et JER vont-ils dans le même sens ?", fontsize=9)
    regime_legend(ax)
    fig.suptitle("VAD oracle, p=0.1, seed 42 : métrique et décomposition de la confusion", fontsize=10)
    save(fig, d, "global_metrique_decomposition.png")


# ============================================================ vues par fichier (p03, p04, p05, p06)
def case(uri, why):
    refs = load_rttm(dk.SAMPLE); ref = refs[uri]
    path = dk.AUDIO / f"{uri}.wav"; dur = sf.info(str(path)).duration
    uem = Timeline([Segment(0, dur)], uri=uri)
    audio = load_audio(str(path), dk.SR); D = FFT(audio, frame_size=dk.FRAME_SIZE, hop_size=dk.HOP_SIZE)
    vad_stack = best_vad(audio, uri=uri, train="devtrain"); del audio
    mfcc = compute_mfcc(D, sr=dk.SR, n_mfcc=dk.N_MFCC); del D
    vad = ref.get_timeline().support().crop(uem)
    X, segs = compute_embeddings(mfcc, vad, win_s=dk.WIN_S, hop_s=dk.HOP_S, min_s=dk.MIN_S, stats=dk.STATS,
                                 drop_c0=dk.DROP_C0, mode=dk.MODE, sr=dk.SR, hop_size=dk.HOP_SIZE)
    n_frames = mfcc.shape[1]; n = len(X); n_ref = len(ref.labels())
    spk, R = dk.ref_frames(ref, n_frames)
    major, share, mixed = dk.window_ref(segs, R)
    si = dk.spectral_internals(X, dk.P_MAIN)
    labs = {"eigengap": dk.labels_for_k(si["V"], si["k_eig"], n), "k oracle": dk.labels_for_k(si["V"], n_ref, n)}
    res = {}
    for name, l in labs.items():
        hyp, post = windows_to_annotation(l, X, segs, vad, n_frames=n_frames, uri=uri, sr=dk.SR, hop_size=dk.HOP_SIZE, return_info=True)
        der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        c = der.compute_components(ref, hyp, uem=uem); mp = der.optimal_mapping(ref, hyp, uem=uem)
        res[name] = (l, post["frame_labels"], mp, c)
    title = f"{uri} — {why} — n_ref={n_ref}, k_eig={si['k_eig']}, {n} windows, p={si['p']} (VAD oracle, seed 42)"
    spk_time = R.sum(0) * dk.FRAME_DUR

    # p03 matrices de confusion
    d = sec("p03_matrices_confusion")
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.1 * n_ref + 8), max(4, .45 * n_ref + 2.5)))
    for ax, (name, (l, fl, mp, c)) in zip(axes, res.items()):
        k = l.max() + 1
        M = np.array([[(R[:, j] & (fl == cc)).sum() * dk.FRAME_DUR for cc in range(k)] for j in range(len(spk))])
        ax.imshow(M, cmap="Blues", aspect="auto")
        for j in range(len(spk)):
            for cc in range(k):
                if M[j, cc] >= .5:
                    hit = mp.get(f"spk{cc:02d}") == spk[j]
                    ax.text(cc, j, f"{M[j, cc]:.0f}", ha="center", va="center", fontsize=6, color="red" if hit else "k",
                            fontweight="bold" if hit else "normal")
        ax.set_yticks(range(len(spk)), [f"{s} ({t:.0f} s){'' if s in mp.values() else '  ✗'}" for s, t in zip(spk, spk_time)], fontsize=6)
        xt = [f"c{cc}→{mp.get(f'spk{cc:02d}', '∅')}" for cc in range(k)]
        ax.set_xticks(range(k), xt, rotation=60, fontsize=6)
        for tl, cc in zip(ax.get_xticklabels(), range(k)):
            if f"spk{cc:02d}" not in mp: tl.set_color("red")
        der_v = (c["false alarm"] + c["missed detection"] + c["confusion"]) / c["total"]
        ax.set_title(f"{name} (k={k}) : secondes réf × cluster ; DER {100 * der_v:.1f} %, confusion {c['confusion']:.0f} s\n"
                     f"rouge gras = case appariée ; c→∅ en rouge = cluster non apparié ; ✗ = locuteur non apparié", fontsize=8)
    fig.suptitle(title, fontsize=9); save(fig, d, f"{uri}_confusion_eig_vs_oracle.png")

    # p04 timeline
    d = sec("p04_timelines")
    fig, ax = plt.subplots(figsize=(18, 3.2 + .18 * n_ref))
    rows = []; y = 0
    ovl = R.sum(1) > 1
    vs = timeline_to_mask(vad_stack, n_frames, sr=dk.SR, hop_size=dk.HOP_SIZE)
    covered = np.zeros(n_frames, bool)
    for s_ in segs:
        covered[int(round(s_.start / dk.FRAME_DUR)):int(round(s_.end / dk.FRAME_DUR))] = True

    def runs(mask):
        ch = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0])); return [(a * dk.FRAME_DUR, (b - a) * dk.FRAME_DUR) for a, b in zip(ch[::2], ch[1::2])]
    speech = R.any(1)
    for name, (l, fl, mp, c) in reversed(list(res.items())):
        inv = {mp[h]: h for h in mp}
        err = np.zeros(n_frames, bool)
        for cc in np.unique(fl[fl >= 0]):
            m = mp.get(f"spk{cc:02d}")
            fr = (fl == cc) & speech
            good = R[:, spk.index(m)] if m is not None else np.zeros(n_frames, bool)
            err |= fr & ~good
        ax.broken_barh(runs(err), (y, .8), color="red"); rows.append((f"erreurs {name}", y)); y += 1
        for cc in range(l.max() + 1):
            ax.broken_barh(runs(fl == cc), (y, .8), color=PAL[(cc + 10) % 20])
        rows.append((f"{name} (k={l.max() + 1})", y)); y += 1.3
    ax.broken_barh(runs(speech & ~covered), (y, .8), color="#000000"); rows.append(("parole sans window", y)); y += 1
    ax.broken_barh(runs(speech & ~vs), (y, .8), color="#999999"); rows.append(("parole hors best_vad devtrain", y)); y += 1
    ax.broken_barh(runs(ovl), (y, .8), color="#882255"); rows.append(("overlap réf.", y)); y += 1.3
    for j, s_ in enumerate(spk):
        ax.broken_barh(runs(R[:, j]), (y, .8), color=PAL[j % 20])
        rows.append((f"{s_} ({spk_time[j]:.0f} s)", y)); y += 1
    ax.set_yticks([b + .4 for _, b in rows], [a for a, _ in rows], fontsize=6); ax.set_xlabel("temps (s)"); ax.set_xlim(0, dur)
    ax.set_title(title + "\nréférence (une ligne par locuteur), overlap, parole hors VAD stack, parole non couverte par une window, "
                 "hypothèses (couleurs = clusters) et trames comptées en confusion (rouge)", fontsize=8)
    save(fig, d, f"{uri}_timeline.png")

    # p05 projections
    d = sec("p05_embeddings")
    P = PCA(2, random_state=0).fit_transform(si["Xn"])
    T = TSNE(2, random_state=0, perplexity=min(30, max(5, n // 5)), init="pca", metric="cosine").fit_transform(si["Xn"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, (proj, pn) in enumerate([(P, "PCA"), (T, "t-SNE")]):
        ax = axes[i, 0]
        ax.scatter(proj[:, 0], proj[:, 1], c=[GREY if (m < 0 or mx) else PAL[m % 20] for m, mx in zip(major, mixed)], s=8)
        ax.legend(handles=[Patch(color=PAL[j % 20], label=f"{s} ({(major == j).sum()} w)") for j, s in enumerate(spk)] + [Patch(color=GREY, label="mixte")], fontsize=6)
        ax.set_title(f"{pn} — vrai locuteur", fontsize=9)
        for j, (name, (l, *_)) in enumerate(res.items()):
            ax = axes[i, j + 1]
            ax.scatter(proj[:, 0], proj[:, 1], c=[PAL[(x + 10) % 20] for x in l], s=8)
            ax.set_title(f"{pn} — cluster {name} (k={l.max() + 1})", fontsize=9)
    fig.suptitle(title + "\nt-SNE (seed 0) déforme les distances globales : ne pas interpréter la distance entre amas", fontsize=9)
    save(fig, d, f"{uri}_projections.png")
    # cosinus intra/inter du fichier
    ok = (major >= 0) & ~mixed
    if ok.sum() > 2 and len(np.unique(major[ok])) > 1:
        Xp, yp = si["Xn"][ok], major[ok]; Sm = Xp @ Xp.T; iu = np.triu_indices(len(yp), 1)
        same = (yp[:, None] == yp[None, :])[iu]; sims = Sm[iu]
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-1, 1, 61)
        ax.hist(sims[same], bins, density=True, alpha=.6, color="#009E73", label=f"intra ({same.sum()} paires)")
        ax.hist(sims[~same], bins, density=True, alpha=.6, color="#D55E00", label=f"inter ({(~same).sum()} paires)")
        ax.set_xlabel("cosinus"); ax.legend(); ax.set_title(f"{uri} : cosinus entre windows pures", fontsize=9)
        save(fig, d, f"{uri}_cosinus_intra_inter.png")

    # p06 affinité et spectre
    d = sec("p06_affinite_spectre")
    order = np.lexsort((np.arange(n), major))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    im = axes[0].imshow(si["A"][np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=axes[0], fraction=.046)
    axes[0].set_title("cosinus brut (avant élagage)", fontsize=9)
    axes[1].imshow(si["A_bin"][np.ix_(order, order)], cmap="Greys", interpolation="nearest")
    axes[1].set_title(f"A_bin après élagage (p={si['p']}) et symétrisation", fontsize=9)
    edges = np.flatnonzero(np.diff(major[order])) + .5
    for ax in axes[:2]:
        for e_ in edges: ax.axhline(e_, color="lime", lw=.5); ax.axvline(e_, color="lime", lw=.5)
        ax.set_xlabel("windows triées par locuteur réf. (majoritaire) puis temps")
    w = si["w"][:min(n, 25)]
    ax = axes[2]; ax.plot(range(1, len(w) + 1), w, "o-", ms=4)
    ax.axvline(si["k_eig"] + .5, color="#777777", ls="--", label=f"coupure eigengap (k={si['k_eig']})")
    ax.axvline(n_ref + .5, color="#009E73", ls=":", lw=2, label=f"n_ref={n_ref}")
    ax.set_xlabel("rang i"); ax.set_ylabel("λ_i (laplacien normalisé)"); ax.legend(fontsize=8)
    ax.set_title("25 premières valeurs propres (eigengap borné à max_speakers=10)", fontsize=9)
    fig.suptitle(title + " ; lignes vertes = changements de locuteur", fontsize=9)
    save(fig, d, f"{uri}_affinite_spectre.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default=",".join(CASES)); ap.add_argument("--skip-cases", action="store_true")
    a = ap.parse_args()
    W, S, SW, EQ, BK, E, F = load()
    p01(W); p02(W, SW, BK); p05_global(W, EQ); p06_global(W); p07_global(S, E); p08_global(E, W)
    p09_global(W, E); p10_global(E, W); p11_global(E, W)
    if not a.skip_cases:
        for u in a.cases.split(","):
            case(u, CASES.get(u, "")); print("cas", u, flush=True)
    print("figures dans", FIG)
