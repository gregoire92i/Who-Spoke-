"""Diagnostic : pourquoi le k oracle dégrade-t-il le DER par rapport à l'eigengap ?
Analyse seulement (aucune modification des sources). Dev sample 90 uniquement.

Pipeline reproduit = HEAD:baseline_oracle.py (celui qui a produit 33,49 % / 38,76 %) :
  frame 65 ms / hop 32.5 ms, MFCC 13, windows 2.0 s / hop 1.0 s / min 0.5 s, mean_std, segment,
  spectral_clustering avec ses valeurs par défaut (p_neighbors=0.1, min_neighbors=5, max_speakers=10,
  standardize=True, seed=42).
VAD : 'oracle' (support de la référence), 'energy' (energy_vad -30 dB, 0.5 / 0.3 s, comme HEAD),
      'stack_devtrain' (best_vad(train='devtrain') : modèle entraîné HORS sample 90, pas de fuite).

Les étapes 1-5 de spectral_clustering sont recopiées dans spectral_internals() pour réutiliser
les valeurs propres entre plusieurs k ; les labels sont vérifiés égaux à ceux de spectral_clustering
(eigengap et oracle) pour p_neighbors=0.1 sur chaque fichier.

Usage : python analysis/clustering_agent/diag_k_oracle.py [--uris a,b,c]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline, Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import (DiarizationErrorRate, JaccardErrorRate,
                                          DiarizationPurity, DiarizationCoverage)
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, roc_auc_score,
                             silhouette_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audio_features import load_audio, FFT, energy_vad, compute_mfcc
from embeddings import compute_embeddings, timeline_to_mask
from clustering import spectral_clustering
from post_treatment import windows_to_annotation
from vad import best_vad

# ---- paramètres (= HEAD:baseline_oracle.py) ----
SR = 16000
FRAME_SIZE = int(65 * SR / 1000)
HOP_SIZE = int(32.5 * SR / 1000)
FRAME_DUR = HOP_SIZE / SR
SEUIL_DB, MIN_SIL, MIN_SPEECH = -30.0, 0.5, 0.3
N_MFCC = 13
WIN_S, HOP_S, MIN_S = 2.0, 1.0, 0.5
STATS, DROP_C0, MODE = "mean_std", False, "segment"
P_MAIN, MIN_NB, MIN_SPK, MAX_SPK, SEED = 0.1, 5, 1, 10, 42
P_EXTRA = [0.05, 0.02]          # balayage diagnostic de p_neighbors (VAD oracle seulement)
MIXED_SHARE = 0.8               # window mixte : locuteur majoritaire < 80 % de la window, ou overlap
COS_BINS = np.linspace(-1, 1, 101)

SAMPLE = "samples/dev/sample90_seed130.rttm"
AUDIO = Path("data/voxconverse/dev/audio")
OUT = Path("results/diag_k_oracle_vs_eigengap")


# ------------------------------------------------------------------ clustering internals
def spectral_internals(X, p_neighbors):
    """Étapes 1 à 5 de clustering.spectral_clustering (copie fidèle). Renvoie dict."""
    n = X.shape[0]
    X = X.astype(float)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    A = Xn @ Xn.T
    p = min(n, max(MIN_NB, int(round(p_neighbors * n))))
    top = np.argsort(-A, axis=1)[:, :p]
    A_bin = np.zeros_like(A)
    np.put_along_axis(A_bin, top, 1.0, axis=1)
    A_bin = 0.5 * (A_bin + A_bin.T)
    deg = A_bin.sum(axis=1)
    d = 1.0 / np.sqrt(deg)
    L = np.eye(n) - d[:, None] * A_bin * d[None, :]
    w, V = np.linalg.eigh(L)
    k_max = min(MAX_SPK, n - 1)
    gaps = np.diff(w[:k_max + 1])
    k_min = min(MIN_SPK, k_max)
    k_eig = int(np.argmax(gaps[k_min - 1:k_max])) + k_min
    return dict(Xn=Xn, A=A, top=top, A_bin=A_bin, w=w, V=V, p=p, k_eig=k_eig)


def labels_for_k(V, k, n):
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=int)
    U = V[:, :k]
    un = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.where(un > 0, un, 1.0)
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(U)


# ------------------------------------------------------------------ référence par frame / window
def ref_frames(ref, n_frames):
    spk = sorted(ref.labels())
    R = np.zeros((n_frames, len(spk)), dtype=bool)
    for j, s in enumerate(spk):
        R[:, j] = timeline_to_mask(ref.label_timeline(s), n_frames, sr=SR, hop_size=HOP_SIZE)
    return spk, R


def window_ref(win_segs, R):
    """Pour chaque window : durée (frames) par locuteur, majoritaire, part, mixte."""
    n_spk = R.shape[1]
    W = np.zeros((len(win_segs), n_spk))
    n_fr = np.zeros(len(win_segs))
    ovl = np.zeros(len(win_segs))
    for i, seg in enumerate(win_segs):
        a = int(round(seg.start / FRAME_DUR)); b = min(int(round(seg.end / FRAME_DUR)), R.shape[0])
        W[i] = R[a:b].sum(axis=0)
        n_fr[i] = max(b - a, 1)
        ovl[i] = (R[a:b].sum(axis=1) > 1).sum()
    major = np.where(W.sum(axis=1) > 0, W.argmax(axis=1), -1)
    share = W.max(axis=1) / n_fr
    mixed = (share < MIXED_SHARE) | (ovl > 0)
    return major, share, mixed


# ------------------------------------------------------------------ évaluation d'une hypothèse
def evaluate(ref, hyp, uem, R, spk, frame_labels, labels, major, mixed):
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    derf = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    pur, cov = DiarizationPurity(), DiarizationCoverage()
    c = der.compute_components(ref, hyp, uem=uem)
    cf = derf.compute_components(ref, hyp, uem=uem)
    cj = jer.compute_components(ref, hyp, uem=uem)
    cp = pur.compute_components(ref, hyp, uem=uem)
    cc = cov.compute_components(ref, hyp, uem=uem)
    mapping = der.optimal_mapping(ref, hyp, uem=uem)       # {"spkXX": ref_label}
    row = {"total_s": c["total"], "FA_s": c["false alarm"], "miss_s": c["missed detection"],
           "conf_s": c["confusion"],
           "DER": (c["false alarm"] + c["missed detection"] + c["confusion"]) / c["total"] if c["total"] else 0,
           "f_total_s": cf["total"], "f_err_s": cf["false alarm"] + cf["missed detection"] + cf["confusion"],
           "DER_forgiving": derf.compute_metric(cf), "jer_count": cj["speaker count"],
           "jer_err": cj["speaker error"], "JER": jer.compute_metric(cj),
           "pur_num": cp["correct"], "pur_den": cp["total"], "purity": pur.compute_metric(cp),
           "cov_num": cc["correct"], "cov_den": cc["total"], "coverage": cov.compute_metric(cc)}
    # --- partition au niveau windows ---
    k = int(labels.max()) + 1 if len(labels) else 0
    sizes = np.bincount(labels, minlength=k) if k else np.array([0])
    row["k_hyp"] = k
    row["largest_cluster_frac"] = sizes.max() / max(sizes.sum(), 1)
    has_ref = major >= 0
    ok = has_ref & ~mixed
    row["frac_win_noref"] = 1 - has_ref.mean() if len(major) else np.nan
    row["frac_win_mixed"] = mixed[has_ref].mean() if has_ref.any() else np.nan
    row["ARI_all"] = adjusted_rand_score(major[has_ref], labels[has_ref]) if has_ref.sum() > 1 else np.nan
    row["ARI_pure"] = adjusted_rand_score(major[ok], labels[ok]) if ok.sum() > 1 else np.nan
    row["AMI_pure"] = adjusted_mutual_info_score(major[ok], labels[ok]) if ok.sum() > 1 else np.nan
    # majoritaire de chaque cluster (windows avec parole de réf.)
    cl_major = {}
    for l in range(k):
        m = (labels == l) & has_ref
        cl_major[l] = int(np.bincount(major[m], minlength=len(spk)).argmax()) if m.any() else -1
    maj = [v for v in cl_major.values() if v >= 0]
    row["n_spk_isolated"] = len(set(maj))                     # locuteurs majoritaires d'au moins 1 cluster
    row["n_clusters_redundant"] = len(maj) - len(set(maj))    # clusters dont le majoritaire l'est déjà ailleurs
    dom = int(R.sum(axis=0).argmax())
    row["n_clusters_dominant"] = sum(v == dom for v in maj)
    # --- décomposition de la confusion au niveau frames ---
    spk_idx = {s: j for j, s in enumerate(spk)}
    speech = R.any(axis=1)
    fl = frame_labels
    lab_ok = fl >= 0
    imp = frag = 0
    for l in range(k):
        mapped = mapping.get(f"spk{l:02d}")
        fr = lab_ok & (fl == l) & speech
        if not fr.any():
            continue
        good = R[fr, spk_idx[mapped]] if mapped is not None else np.zeros(fr.sum(), bool)
        err = (~good).sum() * FRAME_DUR
        if mapped is not None and spk_idx[mapped] == cl_major[l]:
            imp += err                                         # cluster apparié à son majoritaire : impureté
        else:
            frag += err                                        # cluster apparié à un autre / non apparié
    row["conf_impurity_s"] = imp
    row["conf_fragment_s"] = frag
    # rappel par locuteur sous l'appariement du DER
    inv = {v: int(str(h)[3:]) for h, v in mapping.items()}
    rec = {}
    for s in spk:
        t = R[:, spk_idx[s]]
        if s in inv:
            rec[s] = (t & (fl == inv[s])).sum() / max(t.sum(), 1)
        else:
            rec[s] = 0.0
    return row, mapping, rec, cl_major


def merge_by_majority(labels, cl_major):
    """Fusion oracle : chaque cluster prend l'indice de son locuteur majoritaire (diagnostic)."""
    remap = {l: (m if m >= 0 else 10_000 + l) for l, m in cl_major.items()}
    new = np.array([remap[l] for l in labels])
    _, new = np.unique(new, return_inverse=True)
    return new


# ------------------------------------------------------------------ un fichier
def process(uri, ref, rows, spk_rows, emb_rows, hists, check):
    path = AUDIO / f"{uri}.wav"
    dur = sf.info(str(path)).duration
    uem = Timeline([Segment(0, dur)], uri=uri)
    n_ref = len(ref.labels())
    audio = load_audio(str(path), SR)
    D = FFT(audio, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    vads = {"oracle": ref.get_timeline().support().crop(uem),
            "energy": energy_vad(D, sr=SR, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, seuil_db=SEUIL_DB,
                                 min_silence=MIN_SIL, min_speech=MIN_SPEECH, uri=uri),
            "stack_devtrain": best_vad(audio, uri=uri, train="devtrain")}
    del audio
    mfcc = compute_mfcc(D, sr=SR, n_mfcc=N_MFCC)
    del D
    n_frames = mfcc.shape[1]
    spk, R = ref_frames(ref, n_frames)
    ref_time = R.sum(axis=0) * FRAME_DUR

    for vname, vad in vads.items():
        X, win_segs = compute_embeddings(mfcc, vad, win_s=WIN_S, hop_s=HOP_S, min_s=MIN_S, stats=STATS,
                                         drop_c0=DROP_C0, mode=MODE, sr=SR, hop_size=HOP_SIZE)
        n = len(X)
        base = {"uri": uri, "vad": vname, "duration_s": dur, "n_ref_speakers": n_ref, "n_windows": n}
        if n < 2:
            print(f"  {uri} {vname}: {n} window(s), ignoré"); continue
        major, share, mixed = window_ref(win_segs, R)
        vmask = timeline_to_mask(vad, n_frames, sr=SR, hop_size=HOP_SIZE)
        p_list = [P_MAIN] + (P_EXTRA if vname == "oracle" else [])
        for pn in p_list:
            si = spectral_internals(X, pn)
            if check and pn == P_MAIN:
                for orc in (False, True):
                    lab_ref = spectral_clustering(X, oracle=orc, n_speakers=n_ref)
                    k_use = n_ref if orc else si["k_eig"]
                    assert np.array_equal(lab_ref, labels_for_k(si["V"], k_use, n)), f"{uri}: wrapper != clustering.py"
            w = si["w"]
            eig_info = {"p": si["p"], "k_eig": si["k_eig"],
                        "gap_at_keig": w[si["k_eig"]] - w[si["k_eig"] - 1],
                        "gap_at_nref": (w[n_ref] - w[n_ref - 1]) if n_ref < n else np.nan,
                        "lambda_nref": w[n_ref - 1] if n_ref <= n else np.nan,
                        "n_eig_below_0.05": int((w < 0.05).sum())}
            ks = {"eigengap": si["k_eig"], "oracle": n_ref}
            if vname == "oracle" and pn == P_MAIN:          # balayage de k
                for kk in range(1, min(n_ref + 3, n - 1, 20) + 1):
                    ks[f"k={kk}"] = kk
            res = {}
            for mode, kk in ks.items():
                if mode.startswith("k=") and kk in (si["k_eig"], n_ref):
                    pass
                labels = labels_for_k(si["V"], kk, n)
                hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=n_frames, uri=uri,
                                                  sr=SR, hop_size=HOP_SIZE, return_info=True)
                row, mapping, rec, cl_major = evaluate(ref, hyp, uem, R, spk, post["frame_labels"], labels,
                                                       major, mixed)
                rows.append({**base, "p_neighbors": pn, "k_mode": mode, "k_forced": kk, **eig_info, **row,
                             "n_tie": post["n_tie"], "n_uncovered": post["n_uncovered"]})
                res[mode] = (labels, rec, cl_major)
                if mode in ("eigengap", "oracle"):        # fusion oracle des clusters par majoritaire
                    lm = merge_by_majority(labels, cl_major)
                    hm, pm = windows_to_annotation(lm, X, win_segs, vad, n_frames=n_frames, uri=uri,
                                                   sr=SR, hop_size=HOP_SIZE, return_info=True)
                    rm, *_ = evaluate(ref, hm, uem, R, spk, pm["frame_labels"], lm, major, mixed)
                    rows.append({**base, "p_neighbors": pn, "k_mode": mode + "_merged", "k_forced": kk,
                                 **eig_info, **rm, "n_tie": pm["n_tie"], "n_uncovered": pm["n_uncovered"]})
            # --- locuteurs ---
            top = si["top"]
            for j, s in enumerate(spk):
                mine = np.flatnonzero(major == j)
                if len(mine):
                    nb = top[mine, 1:]                         # voisins hors soi-même (approx : self en tête)
                    nb_pur = (major[nb] == j).mean()
                else:
                    nb_pur = np.nan
                le, ce = res["eigengap"][1], res["eigengap"][2]
                lo, co = res["oracle"][1], res["oracle"][2]
                spk_rows.append({"uri": uri, "vad": vname, "p_neighbors": pn, "speaker": s, "n_ref_speakers": n_ref,
                                 "ref_time_s": ref_time[j], "ref_share": ref_time[j] / ref_time.sum(),
                                 "time_in_vad_s": (R[:, j] & vmask).sum() * FRAME_DUR,
                                 "n_win_major": len(mine), "n_win_pure": int(((major == j) & ~mixed).sum()),
                                 "p": si["p"], "n_windows": n, "neighbor_purity": nb_pur,
                                 "recall_eig": le[s], "recall_oracle": lo[s],
                                 "isolated_eig": j in ce.values(), "isolated_oracle": j in co.values(),
                                 "is_dominant": j == int(ref_time.argmax())})
            # --- qualité intrinsèque des embeddings (une fois par VAD) ---
            if pn == P_MAIN:
                ok = (major >= 0) & ~mixed
                e = {"uri": uri, "vad": vname, "n_ref_speakers": n_ref, "n_windows": n, "n_pure": int(ok.sum())}
                if ok.sum() > 2 and len(np.unique(major[ok])) > 1:
                    Xp, yp = si["Xn"][ok], major[ok]
                    S = Xp @ Xp.T
                    iu = np.triu_indices(len(yp), 1)
                    same = (yp[:, None] == yp[None, :])[iu]
                    sims = S[iu]
                    e["auc_same_vs_diff"] = roc_auc_score(same, sims)
                    e["cos_intra_mean"], e["cos_inter_mean"] = sims[same].mean(), sims[~same].mean()
                    e["silhouette_ref"] = silhouette_score(Xp, yp, metric="cosine")
                    for mode in ("eigengap", "oracle"):
                        lab = res[mode][0][ok]
                        e[f"silhouette_{mode}"] = silhouette_score(Xp, lab, metric="cosine") if len(np.unique(lab)) > 1 else np.nan
                    if vname == "oracle":
                        hists["intra"] += np.histogram(sims[same], COS_BINS)[0]
                        hists["inter"] += np.histogram(sims[~same], COS_BINS)[0]
                        # Fisher par dimension (embedding brut standardisé, avant L2)
                        Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
                        Xs = Xs[ok]
                        mu = Xs.mean(0)
                        sb = sum((yp == c).sum() * (Xs[yp == c].mean(0) - mu) ** 2 for c in np.unique(yp))
                        sw = sum(((Xs[yp == c] - Xs[yp == c].mean(0)) ** 2).sum(0) for c in np.unique(yp))
                        for d_, f in enumerate(sb / np.maximum(sw, 1e-12)):
                            e[f"fisher_d{d_:02d}"] = f
                emb_rows.append(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uris", default=None)
    ap.add_argument("--tag", default="sample90")
    args = ap.parse_args()
    refs = load_rttm(SAMPLE)
    uris = sorted(refs) if args.uris is None else args.uris.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, spk_rows, emb_rows = [], [], []
    hists = {"intra": np.zeros(len(COS_BINS) - 1), "inter": np.zeros(len(COS_BINS) - 1)}
    t0 = time.time()
    for i, uri in enumerate(uris, 1):
        process(uri, refs[uri], rows, spk_rows, emb_rows, hists, check=True)
        print(f"[{i}/{len(uris)}] {uri} ({time.time() - t0:.0f} s)", flush=True)
        # sauvegarde incrémentale
        pd.DataFrame(rows).to_csv(OUT / f"runs_{args.tag}.csv", index=False)
        pd.DataFrame(spk_rows).to_csv(OUT / f"speakers_{args.tag}.csv", index=False)
        pd.DataFrame(emb_rows).to_csv(OUT / f"embeddings_quality_{args.tag}.csv", index=False)
        np.savez(OUT / f"cos_hist_{args.tag}.npz", bins=COS_BINS, **hists)


if __name__ == "__main__":
    main()
