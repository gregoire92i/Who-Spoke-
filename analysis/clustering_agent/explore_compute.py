"""Calculs complémentaires pour les figures exploratoires (k oracle vs eigengap). Dev sample 90 uniquement.
Réutilise diag_k_oracle.py (mêmes paramètres = HEAD:baseline_oracle.py ; spectral_clustering par défaut).

Pour chaque fichier (un audio à la fois) :
  - VAD oracle / energy / stack_devtrain, p=0.1, seed 42 : eigengap, oracle, et versions « regroupées » ;
  - VAD oracle seulement : seeds KMeans {0,1,2,3,42}, p_neighbors {0.02, 0.05, 0.1, 0.15, 0.2} (seed 42).
Pour chaque hypothèse, la confusion est recomptée trame par trame (trames de 32,5 ms) et ventilée :
  - par type de cluster : impureté (cluster apparié à son majoritaire) / mal apparié (apparié à un autre locuteur)
    / non apparié (aucun locuteur associé par optimal_mapping) ;
  - par position : overlap de référence ; distance à la frontière de tour de parole la plus proche
    (0-0.5, 0.5-1, 1-2, >2 s) ; trame « disputée » (couverte par des windows de labels différents) ou non.
Sortie : results/diag_k_oracle_vs_eigengap/explore_runs.csv, explore_files.csv
Usage : python analysis/clustering_agent/explore_compute.py [--uris a,b]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_k_oracle as dk
from audio_features import load_audio, FFT, energy_vad, compute_mfcc
from embeddings import compute_embeddings, timeline_to_mask
from post_treatment import windows_to_annotation
from vad import best_vad

SEEDS = [0, 1, 2, 3, 42]
P_SWEEP = [0.02, 0.05, 0.1, 0.15, 0.2]
BND_BINS = [0, 0.5, 1.0, 2.0, np.inf]
BND_NAMES = ["0-0.5s", "0.5-1s", "1-2s", ">2s"]


def boundary_distance(ref, n_frames):
    """Distance (s) de chaque trame à la frontière de tour de parole la plus proche (début/fin d'un segment)."""
    pts = np.array(sorted({t for seg, _, _ in ref.itertracks(yield_label=True) for t in (seg.start, seg.end)}))
    t = np.arange(n_frames) * dk.FRAME_DUR
    if len(pts) == 0:
        return np.full(n_frames, np.inf)
    idx = np.clip(np.searchsorted(pts, t), 1, len(pts) - 1) if len(pts) > 1 else np.zeros(n_frames, int)
    return np.minimum(np.abs(t - pts[idx]), np.abs(t - pts[np.maximum(idx - 1, 0)]))


def disputed_mask(labels, win_segs, n_frames):
    k = int(labels.max()) + 1
    cover = np.zeros((n_frames, k), dtype=bool)
    for seg, l in zip(win_segs, labels):
        a = int(round(seg.start / dk.FRAME_DUR)); b = min(int(round(seg.end / dk.FRAME_DUR)), n_frames)
        cover[a:b, l] = True
    return cover.sum(axis=1) > 1


def decompose(ref, R, spk, fl, mapping, cl_major, labels, win_segs, bdist):
    n_frames = len(fl)
    idx = {s: j for j, s in enumerate(spk)}
    speech = R.any(axis=1)
    lab = fl >= 0
    err = np.zeros(n_frames, bool); typ = np.full(n_frames, "", dtype=object)
    for l in np.unique(fl[lab]):
        m = mapping.get(f"spk{l:02d}")
        fr = (fl == l) & speech
        good = R[fr, idx[m]] if m is not None else np.zeros(fr.sum(), bool)
        e = np.zeros(n_frames, bool); e[np.flatnonzero(fr)[~good]] = True
        err |= e
        typ[e] = "unmapped" if m is None else ("impurity" if idx[m] == cl_major.get(int(l), -2) else "mismapped")
    ovl = R.sum(axis=1) > 1
    disp = disputed_mask(labels, win_segs, n_frames) if len(labels) else np.zeros(n_frames, bool)
    d = {}
    for t in ["impurity", "mismapped", "unmapped"]:
        d[f"cf_{t}_s"] = (typ == t).sum() * dk.FRAME_DUR
    d["cf_frames_s"] = err.sum() * dk.FRAME_DUR
    d["cf_overlap_s"] = (err & ovl).sum() * dk.FRAME_DUR
    d["speech_overlap_s"] = (speech & ovl).sum() * dk.FRAME_DUR
    for (lo, hi), name in zip(zip(BND_BINS[:-1], BND_BINS[1:]), BND_NAMES):
        m = (bdist >= lo) & (bdist < hi) & speech & ~ovl
        d[f"cf_bnd_{name}_s"] = (err & m).sum() * dk.FRAME_DUR
        d[f"speech_bnd_{name}_s"] = m.sum() * dk.FRAME_DUR
    d["cf_disputed_s"] = (err & disp).sum() * dk.FRAME_DUR
    d["speech_disputed_s"] = (speech & disp & lab).sum() * dk.FRAME_DUR
    d["cf_undisputed_s"] = (err & ~disp).sum() * dk.FRAME_DUR
    d["speech_labeled_s"] = (speech & lab).sum() * dk.FRAME_DUR
    # locuteurs de référence non appariés et leur temps
    mapped_ref = set(mapping.values())
    lost = [j for j, s in enumerate(spk) if s not in mapped_ref]
    d["n_ref_unmapped"] = len(lost)
    d["time_ref_unmapped_s"] = R[:, lost].sum() * dk.FRAME_DUR if lost else 0.0
    return d


def run_one(ref, uem, R, spk, X, win_segs, vad, n_frames, labels, major, mixed, bdist, uri):
    hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=n_frames, uri=uri, sr=dk.SR,
                                      hop_size=dk.HOP_SIZE, return_info=True)
    row, mapping, rec, cl_major = dk.evaluate(ref, hyp, uem, R, spk, post["frame_labels"], labels, major, mixed)
    row.update(decompose(ref, R, spk, post["frame_labels"], mapping, cl_major, labels, win_segs, bdist))
    row["n_tie"] = post["n_tie"]
    return row, cl_major


def process(uri, ref, rows, frows):
    path = dk.AUDIO / f"{uri}.wav"; dur = sf.info(str(path)).duration
    uem = Timeline([Segment(0, dur)], uri=uri); n_ref = len(ref.labels())
    audio = load_audio(str(path), dk.SR); D = FFT(audio, frame_size=dk.FRAME_SIZE, hop_size=dk.HOP_SIZE)
    vads = {"oracle": ref.get_timeline().support().crop(uem),
            "energy": energy_vad(D, sr=dk.SR, frame_size=dk.FRAME_SIZE, hop_size=dk.HOP_SIZE, seuil_db=dk.SEUIL_DB,
                                 min_silence=dk.MIN_SIL, min_speech=dk.MIN_SPEECH, uri=uri),
            "stack_devtrain": best_vad(audio, uri=uri, train="devtrain")}
    del audio
    mfcc = compute_mfcc(D, sr=dk.SR, n_mfcc=dk.N_MFCC); del D
    n_frames = mfcc.shape[1]
    spk, R = dk.ref_frames(ref, n_frames)
    bdist = boundary_distance(ref, n_frames)
    speech = R.any(axis=1)
    frows.append({"uri": uri, "duration_s": dur, "n_ref_speakers": n_ref, "speech_s": speech.sum() * dk.FRAME_DUR,
                  "overlap_rate": (R.sum(1) > 1).sum() / max(speech.sum(), 1),
                  "dom_share": R.sum(0).max() / max(R.sum(), 1),
                  "n_turns": sum(1 for _ in ref.itertracks()),
                  "mean_turn_s": float(np.mean([s.duration for s in ref.itersegments()]))})
    for vname, vad in vads.items():
        X, win_segs = compute_embeddings(mfcc, vad, win_s=dk.WIN_S, hop_s=dk.HOP_S, min_s=dk.MIN_S, stats=dk.STATS,
                                         drop_c0=dk.DROP_C0, mode=dk.MODE, sr=dk.SR, hop_size=dk.HOP_SIZE)
        n = len(X)
        if n < 2:
            continue
        major, share, mixed = dk.window_ref(win_segs, R)
        p_list = P_SWEEP if vname == "oracle" else [dk.P_MAIN]
        for pn in p_list:
            si = dk.spectral_internals(X, pn)
            seeds = SEEDS if (vname == "oracle" and pn == dk.P_MAIN) else [42]
            for seed in seeds:
                dk.SEED = seed                                   # labels_for_k lit dk.SEED
                for mode, kk in (("eigengap", si["k_eig"]), ("oracle", n_ref)):
                    labels = dk.labels_for_k(si["V"], kk, n)
                    row, cl_major = run_one(ref, uem, R, spk, X, win_segs, vad, n_frames, labels, major, mixed, bdist, uri)
                    base = {"uri": uri, "vad": vname, "p_neighbors": pn, "seed": seed, "n_ref_speakers": n_ref,
                            "n_windows": n, "p": si["p"], "k_eig": si["k_eig"]}
                    rows.append({**base, "k_mode": mode, **row})
                    if seed == 42 and pn == dk.P_MAIN:
                        lm = dk.merge_by_majority(labels, cl_major)
                        rm, _ = run_one(ref, uem, R, spk, X, win_segs, vad, n_frames, lm, major, mixed, bdist, uri)
                        rows.append({**base, "k_mode": mode + "_merged", **rm})
            dk.SEED = 42


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--uris", default=None); ap.add_argument("--tag", default="sample90")
    a = ap.parse_args()
    refs = load_rttm(dk.SAMPLE)
    uris = sorted(refs) if a.uris is None else a.uris.split(",")
    rows, frows = [], []; t0 = time.time()
    for i, u in enumerate(uris, 1):
        process(u, refs[u], rows, frows)
        print(f"[{i}/{len(uris)}] {u} ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(dk.OUT / f"explore_runs_{a.tag}.csv", index=False)
        pd.DataFrame(frows).to_csv(dk.OUT / f"explore_files_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
