"""Figures du diagnostic k oracle vs eigengap (backend Agg, jamais plt.show()).
Usage : python analysis/clustering_agent/figures_k_oracle.py [--cases uri1,uri2]
Seeds : KMeans 42 (comme clustering.py), t-SNE random_state=0."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from embeddings import compute_embeddings
from post_treatment import windows_to_annotation

OUT = dk.OUT
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TAG = "sample90"
# palette Okabe-Ito (daltoniens) + compléments
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000",
       "#999933", "#882255", "#44AA99", "#117733", "#332288", "#AA4499", "#88CCEE", "#DDCC77",
       "#661100", "#6699CC", "#AA4466", "#4477AA"]
GREY = "#BBBBBB"
REG_COL = {"k_eig<n_ref": "#D55E00", "k_eig=n_ref": "#999999", "k_eig>n_ref": "#0072B2"}


def fig_delta(w):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, v in enumerate(["energy", "oracle"]):
        x = w[w.vad == v].sort_values("dconf_s").reset_index(drop=True)
        ax = axes[row, 0]
        ax.bar(range(len(x)), x.dconf_s, color=[REG_COL[g] for g in x.regime])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"VAD {v} : Δ confusion (s) par fichier, k oracle − eigengap\n"
                     f"total {100 * x.dconf_s.sum() / x.total_s_eig.sum():+.2f} pts de DER")
        ax.set_xlabel("fichiers triés"); ax.set_ylabel("Δ confusion (s)")
        for g, c in REG_COL.items():
            ax.bar([], [], color=c, label=f"{g} (n={(x.regime == g).sum()})")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.scatter(x.k_err_eig + np.random.default_rng(0).uniform(-.15, .15, len(x)), 100 * x.dDER,
                   c=[REG_COL[g] for g in x.regime], s=18 + 2 * x.n_ref_speakers, alpha=.8)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("k_eigengap − n_ref (jitter horizontal ±0,15 ; taille ∝ n_ref)")
        ax.set_ylabel("ΔDER moderne (pts)")
        ax.set_title(f"VAD {v} : ΔDER par fichier selon l'erreur de l'eigengap")
    fig.suptitle("Fixe : 90 fichiers dev, embeddings MFCC 2 s / 1 s, p_neighbors=0.1, seed 42. Varie : k.", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "delta_confusion_par_fichier.png", dpi=150); plt.close(fig)


def fig_pur_cov(w):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, v in zip(axes, ["energy", "oracle"]):
        x = w[(w.vad == v) & (w.k_hyp_eig != w.k_hyp_orc)]
        for _, r in x.iterrows():
            ax.annotate("", xy=(r.coverage_orc, r.purity_orc), xytext=(r.coverage_eig, r.purity_eig),
                        arrowprops=dict(arrowstyle="->", color=REG_COL[r.regime], alpha=.7, lw=1))
        for g, c in REG_COL.items():
            ax.plot([], [], color=c, label=f"{g} (n={(x.regime == g).sum()})")
        ax.set_xlabel("coverage (fichier)"); ax.set_ylabel("purity (fichier)")
        ax.set_xlim(0.2, 1.02); ax.set_ylim(0.2, 1.02); ax.legend(fontsize=8)
        ax.set_title(f"VAD {v} : flèche = eigengap → k oracle\n(fichiers où k change)")
    fig.suptitle("Purity / coverage par fichier : le k oracle déplace-t-il vers plus de purity ou moins de coverage ?", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "purity_coverage_eig_vers_oracle.png", dpi=150); plt.close(fig)


def fig_speakers(s):
    sp = s[(s.vad == "oracle") & (s.p_neighbors == 0.1) & (s.n_ref_speakers >= 2)].copy()
    sp["x"] = (sp.n_win_major + 0.5) / sp.p
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    nd = sp[~sp.is_dominant]
    ax.scatter(nd.x, nd.neighbor_purity, s=10, alpha=.5, color="#0072B2", label="non dominants")
    d = sp[sp.is_dominant]
    ax.scatter(d.x, d.neighbor_purity, s=10, alpha=.6, color="#D55E00", label="dominant du fichier")
    xx = np.logspace(-2, 2, 200)
    ax.plot(xx, np.minimum(1, xx), "k--", lw=.8, label="plafond ≈ min(1, n_win/p)")
    ax.set_xscale("log"); ax.set_xlabel("(n windows majoritaires + 0,5) / p"); ax.set_ylabel("neighbor purity")
    ax.set_title("Part des p voisins d'une window\nqui appartiennent au même locuteur"); ax.legend(fontsize=7)
    ax = axes[1]
    bins = [0, 0.01, 0.5, 1, 2, 5, 1e9]; lab = ["0", "<0.5", "0.5–1", "1–2", "2–5", ">5"]
    sp["bin"] = pd.cut(sp.n_win_major / sp.p, bins, labels=lab, include_lowest=True, right=False)
    g = sp.groupby("bin", observed=False)
    xs = np.arange(len(lab))
    ax.bar(xs - .2, g.isolated_eig.mean(), .4, color="#999999", label="eigengap")
    ax.bar(xs + .2, g.isolated_oracle.mean(), .4, color="#009E73", label="k oracle")
    for i, n in enumerate(g.size()):
        ax.text(i, 1.02, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(xs, lab); ax.set_xlabel("n windows majoritaires / p"); ax.set_ylim(0, 1.1)
    ax.set_ylabel("part des locuteurs majoritaires d'un cluster"); ax.legend(fontsize=8)
    ax.set_title("Un locuteur obtient-il « son » cluster ?")
    ax = axes[2]
    ax.bar(xs - .2, g.recall_eig.mean(), .4, color="#999999", label="eigengap")
    ax.bar(xs + .2, g.recall_oracle.mean(), .4, color="#009E73", label="k oracle")
    ax.set_xticks(xs, lab); ax.set_xlabel("n windows majoritaires / p"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("rappel moyen sous l'appariement du DER"); ax.legend(fontsize=8)
    ax.set_title("Temps du locuteur correctement attribué")
    fig.suptitle("VAD oracle, p_neighbors=0.1, 90 fichiers dev (locuteurs des fichiers à n_ref ≥ 2)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "locuteurs_rares_isolation.png", dpi=150); plt.close(fig)


def fig_sweep(sw):
    x = sw[sw.n_ref_speakers >= 2]
    ks = sorted(x.k_minus_nref.unique()); ks = [k for k in ks if (x.k_minus_nref == k).sum() >= 5]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, col, name in zip(axes, ["DER", "purity", "coverage", "ARI_pure"],
                             ["DER moderne", "purity", "coverage", "ARI (windows pures)"]):
        data = [x[x.k_minus_nref == k][col].dropna().values for k in ks]
        ax.boxplot(data, positions=range(len(ks)), widths=.6, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(i + np.random.default_rng(i).uniform(-.2, .2, len(d)), d, s=5, alpha=.35, color="#0072B2")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks]); ax.set_xlabel("k forcé − n_ref")
        ax.set_title(name); ax.axvline(ks.index(0), color="#D55E00", ls="--", lw=.8)
        if col == "DER": ax.set_ylim(0, 1.2)
    fig.suptitle("Balayage de k (VAD oracle, p_neighbors=0.1, seed 42), fichiers n_ref ≥ 2 ; points = fichiers, "
                 "n fichiers varie avec k − n_ref", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "sweep_k_purity_coverage_DER.png", dpi=150); plt.close(fig)


def fig_cos(eq):
    h = np.load(OUT / f"cos_hist_{TAG}.npz")
    c = 0.5 * (h["bins"][1:] + h["bins"][:-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(c, h["intra"] / h["intra"].sum(), color="#009E73", label="intra-locuteur")
    ax.plot(c, h["inter"] / h["inter"].sum(), color="#D55E00", label="inter-locuteurs")
    ax.set_xlabel("similarité cosinus (embeddings centrés-réduits par fichier, L2)"); ax.set_ylabel("densité (somme = 1)")
    ax.set_title("Paires de windows pures, même fichier, 90 fichiers poolés"); ax.legend()
    ax = axes[1]
    x = eq[(eq.vad == "oracle")].dropna(subset=["auc_same_vs_diff"])
    ax.scatter(x.n_ref_speakers, x.auc_same_vs_diff, s=15, color="#0072B2")
    ax.set_xlabel("n_ref"); ax.set_ylabel("AUC même / différent locuteur (par fichier)")
    ax.set_title(f"AUC médiane {x.auc_same_vs_diff.median():.3f} ({len(x)} fichiers)")
    fig.suptitle("VAD oracle : séparabilité intrinsèque des embeddings MFCC (avant clustering)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "cosinus_intra_inter.png", dpi=150); plt.close(fig)


def case_study(uri):
    refs = load_rttm(dk.SAMPLE); ref = refs[uri]
    path = dk.AUDIO / f"{uri}.wav"; dur = sf.info(str(path)).duration
    uem = Timeline([Segment(0, dur)], uri=uri)
    audio = load_audio(str(path), dk.SR); D = FFT(audio, frame_size=dk.FRAME_SIZE, hop_size=dk.HOP_SIZE); del audio
    mfcc = compute_mfcc(D, sr=dk.SR, n_mfcc=dk.N_MFCC); del D
    vad = ref.get_timeline().support().crop(uem)
    X, segs = compute_embeddings(mfcc, vad, win_s=dk.WIN_S, hop_s=dk.HOP_S, min_s=dk.MIN_S, stats=dk.STATS,
                                 drop_c0=dk.DROP_C0, mode=dk.MODE, sr=dk.SR, hop_size=dk.HOP_SIZE)
    n_frames = mfcc.shape[1]; n = len(X); n_ref = len(ref.labels())
    spk, R = dk.ref_frames(ref, n_frames)
    major, share, mixed = dk.window_ref(segs, R)
    si = dk.spectral_internals(X, dk.P_MAIN)
    lab = {"eigengap": dk.labels_for_k(si["V"], si["k_eig"], n), "k oracle": dk.labels_for_k(si["V"], n_ref, n)}
    order = np.lexsort((np.arange(n), major))
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 1.1, 1, 0.55])
    # affinité binarisée triée par locuteur de réf puis temps
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(si["A_bin"][np.ix_(order, order)], cmap="Greys", interpolation="nearest")
    edges = np.flatnonzero(np.diff(major[order])) + 0.5
    for e_ in edges: ax.axhline(e_, color="r", lw=.5); ax.axvline(e_, color="r", lw=.5)
    ax.set_title(f"A_bin (p={si['p']} voisins / {n} windows)\ntrié par locuteur de réf. puis temps", fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    w = si["w"][:min(n, 21)]
    ax.plot(range(1, len(w) + 1), w, "o-", ms=4)
    ax.axvline(si["k_eig"] + .5, color="#999999", ls="--", label=f"gap eigengap k={si['k_eig']}")
    ax.axvline(n_ref + .5, color="#009E73", ls=":", label=f"n_ref={n_ref}")
    ax.set_xlabel("rang i"); ax.set_ylabel("λ_i (laplacien normalisé)"); ax.legend(fontsize=8)
    ax.set_title("Spectre", fontsize=9)
    # projections
    ok = major >= 0
    P = PCA(2, random_state=0).fit_transform(si["Xn"])
    T = TSNE(2, random_state=0, perplexity=min(30, max(5, n // 5)), init="pca", metric="cosine").fit_transform(si["Xn"])
    for j, (proj, name) in enumerate([(P, "PCA"), (T, "t-SNE")]):
        ax = fig.add_subplot(gs[0, 2 + j])
        cols = [GREY if (m < 0 or mx) else PAL[m % 20] for m, mx in zip(major, mixed)]
        ax.scatter(proj[:, 0], proj[:, 1], c=cols, s=8)
        for jj, s_ in enumerate(spk):
            ax.scatter([], [], color=PAL[jj % 20], label=f"{s_} ({(major == jj).sum()} w)")
        ax.legend(fontsize=6, markerscale=1.5); ax.set_title(f"{name} — couleur = locuteur de réf. (gris = mixte)", fontsize=9)
    for i, (name, l) in enumerate(lab.items()):
        for j, (proj, pn) in enumerate([(P, "PCA"), (T, "t-SNE")]):
            ax = fig.add_subplot(gs[1, 2 * i + j])
            ax.scatter(proj[:, 0], proj[:, 1], c=[PAL[(x + 10) % 20] for x in l], s=8)
            ax.set_title(f"{pn} — couleur = cluster ({name}, k={l.max() + 1})", fontsize=9)
    # matrices de confusion (s) et timelines
    tl_ax = fig.add_subplot(gs[3, :])
    yt = []
    for jj, s_ in enumerate(spk):
        for seg in ref.label_timeline(s_):
            tl_ax.broken_barh([(seg.start, seg.duration)], (0, 0.8), color=PAL[jj % 20])
    yt.append(("référence", 0.4))
    for i, (name, l) in enumerate(lab.items()):
        hyp, post = windows_to_annotation(l, X, segs, vad, n_frames=n_frames, uri=uri, sr=dk.SR,
                                          hop_size=dk.HOP_SIZE, return_info=True)
        der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        mapping = der.optimal_mapping(ref, hyp, uem=uem)
        c = der.compute_components(ref, hyp, uem=uem)
        ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2])
        k = l.max() + 1
        M = np.zeros((len(spk), k))
        fl = post["frame_labels"]
        for jj in range(len(spk)):
            for cc in range(k):
                M[jj, cc] = (R[:, jj] & (fl == cc)).sum() * dk.FRAME_DUR
        ax.imshow(M, cmap="Blues", aspect="auto")
        for jj in range(len(spk)):
            for cc in range(k):
                ax.text(cc, jj, f"{M[jj, cc]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(spk)), [f"{s_} ({R[:, jj].sum() * dk.FRAME_DUR:.0f}s)" for jj, s_ in enumerate(spk)], fontsize=7)
        ax.set_xticks(range(k), [f"c{cc}→{mapping.get(f'spk{cc:02d}', '∅')}" for cc in range(k)], fontsize=7, rotation=30)
        derv = (c["false alarm"] + c["missed detection"] + c["confusion"]) / c["total"]
        ax.set_title(f"{name} (k={k}) : secondes réf × cluster ; DER {100 * derv:.1f} %, "
                     f"confusion {c['confusion']:.0f} s", fontsize=9)
        y0 = 1 + i
        for cc in range(k):
            m = fl == cc
            ch = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
            for a, b in zip(ch[::2], ch[1::2]):
                tl_ax.broken_barh([(a * dk.FRAME_DUR, (b - a) * dk.FRAME_DUR)], (y0, 0.8), color=PAL[(cc + 10) % 20])
        yt.append((name, y0 + .4))
    tl_ax.set_yticks([y for _, y in yt], [t for t, _ in yt]); tl_ax.set_xlabel("temps (s)")
    tl_ax.set_title("Timeline : référence (couleurs locuteurs) ; hypothèses (couleurs clusters, palette différente)", fontsize=9)
    fig.suptitle(f"{uri} — VAD oracle, MFCC 2 s / 1 s, p_neighbors=0.1, KMeans seed 42, t-SNE seed 0 "
                 f"(t-SNE déforme les distances globales : ne pas lire la distance entre amas)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"{uri}_cas_k_oracle.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="")
    a = ap.parse_args()
    w = pd.read_csv(OUT / f"par_fichier_eig_vs_oracle_{TAG}.csv")
    s = pd.read_csv(OUT / f"speakers_{TAG}.csv")
    sw = pd.read_csv(OUT / f"sweep_k_{TAG}.csv")
    eq = pd.read_csv(OUT / f"embeddings_quality_{TAG}.csv")
    fig_delta(w); fig_pur_cov(w); fig_speakers(s); fig_sweep(sw); fig_cos(eq)
    for u in [u for u in a.cases.split(",") if u]:
        case_study(u)
    print("figures dans", FIG)
