"""Expérience NME-SC (amélioration n°2, clustering) : p_neighbors choisi fichier par fichier.

Compare, sur LES MÊMES embeddings de chaque fichier (une seule VAD, un seul calcul de MFCC/embeddings) :
  - "fixe_p0.02"        : spectral_clustering avec p_neighbors fixe (valeur actuelle de baseline.py) ;
  - "nme_normalized"    : nme_spectral_clustering, Laplacien normalisé (seul le choix de p change) ;
  - "nme_unnormalized"  : nme_spectral_clustering, Laplacien non normalisé (comme l'article).
Tous les autres paramètres sont lus dans baseline.py (ils y font foi).

VAD : best_vad(train="devtrain") -> modèles entraînés HORS sample90 : pas de fuite sur le dev sample 90.
Protocole D-30 à D-37 : collar 0, UEM [0, durée du wav], micro, DER moderne / forgiving / JER + FA, missed, confusion.
Dev uniquement (Dev = régler). Un fichier audio à la fois.

Usage : python nme_sc_experiment.py [--rttm samples/dev/sample90_seed130.rttm] [--limit N]"""
import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

import baseline as B  # paramètres de la baseline (main() n'est pas exécuté à l'import)
from audio_features import load_audio, FFT, compute_mfcc
from vad import best_vad
from embeddings import compute_embeddings
from clustering import spectral_clustering, nme_spectral_clustering
from post_treatment import windows_to_annotation

VAD_TRAIN = "devtrain"
NME_PARAMS = dict(min_neighbors=B.min_neighbors, max_frac=0.25, n_grid=30)
OUT_DIR = Path("results/nme_sc")


def arms():
    """Variantes de clustering comparées : nom -> fonction(X) -> (labels, info)."""
    common = dict(oracle=False, min_speakers=B.min_speakers, max_speakers=B.max_speakers,
                  standardize=B.standardize, seed=B.seed, return_info=True)
    return {
        f"fixe_p{B.p_neighbors}": lambda X: spectral_clustering(
            X, p_neighbors=B.p_neighbors, min_neighbors=B.min_neighbors, **common),
        "nme_normalized": lambda X: nme_spectral_clustering(X, laplacian="normalized", **NME_PARAMS, **common),
        "nme_unnormalized": lambda X: nme_spectral_clustering(X, laplacian="unnormalized", **NME_PARAMS, **common),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm", default="samples/dev/sample90_seed130.rttm")
    ap.add_argument("--audio_dir", default="data/voxconverse/dev/audio")
    ap.add_argument("--limit", type=int, default=None, help="n premiers fichiers (test rapide)")
    args = ap.parse_args()
    if "test" in args.rttm:
        raise SystemExit("Dev uniquement : le test ne sert pas à régler (CLAUDE.md, section 3).")
    audio_dir = Path(args.audio_dir)
    tag = Path(args.rttm).stem + (f"_limit{args.limit}" if args.limit else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations = {u: a for u, a in load_rttm(args.rttm).items() if (audio_dir / f"{u}.wav").exists()}
    uris = sorted(annotations)[:args.limit] if args.limit else sorted(annotations)

    variants = arms()
    metrics = {name: dict(der=DiarizationErrorRate(collar=0.0, skip_overlap=False),
                          derf=DiarizationErrorRate(collar=0.0, skip_overlap=True),
                          jer=JaccardErrorRate(collar=0.0, skip_overlap=False)) for name in variants}
    det = DetectionErrorRate(collar=0.0, skip_overlap=False)
    rows = {name: [] for name in variants}
    rttm_files = {name: open(OUT_DIR / f"{name}_{tag}.rttm", "w") for name in variants}

    for i, uri in enumerate(uris, start=1):
        ref = annotations[uri]
        path = audio_dir / f"{uri}.wav"
        duration = sf.info(str(path)).duration
        uem = Timeline([Segment(0, duration)], uri=uri)

        # --- partie commune à toutes les variantes ---
        t0 = time.perf_counter()
        audio = load_audio(str(path), B.sr)
        D = FFT(audio, frame_size=B.frame_size, hop_size=B.hop_size)
        vad = best_vad(audio, uri=uri, train=VAD_TRAIN)
        del audio
        mfcc = compute_mfcc(D, sr=B.sr, n_mfcc=B.n_mfcc)
        del D
        X, win_segs = compute_embeddings(mfcc, vad, win_s=B.win_s, hop_s=B.hop_s, min_s=B.min_s, stats=B.stats,
                                         drop_c0=B.drop_c0, mode=B.mode, sr=B.sr, hop_size=B.hop_size)
        t_common = time.perf_counter() - t0
        vad_err = det(ref, vad.to_annotation(), uem=uem)

        line = f"[{i}/{len(uris)}] {uri} (réf {len(ref.labels())})"
        for name, fn in variants.items():
            t1 = time.perf_counter()
            labels, info = fn(X)
            hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                              sr=B.sr, hop_size=B.hop_size, return_info=True)
            t_clust = time.perf_counter() - t1
            m = metrics[name]
            c = m["der"](ref, hyp, uem=uem, detailed=True)
            row = {
                "uri": uri, "duration_s": duration, "n_ref_speakers": len(ref.labels()),
                "k_hyp": info["k"], "p_chosen": info.get("p", None), "n_windows": len(X),
                "DER": c["diarization error rate"],
                "DER_forgiving": m["derf"](ref, hyp, uem=uem),
                "JER": m["jer"](ref, hyp, uem=uem),
                "VAD_error": vad_err,
                "FA_s": c["false alarm"], "missed_s": c["missed detection"], "confusion_s": c["confusion"],
                "total_s": c["total"],
                "RTF": (t_common + t_clust) / duration, "t_clustering_s": t_clust,
            }
            rows[name].append(row)
            hyp.write_rttm(rttm_files[name])
            line += f" | {name}: DER {row['DER']:.1%} k {row['k_hyp']}"
            if row["p_chosen"] is not None:
                line += f" p {row['p_chosen']}"
        print(line, flush=True)
        del mfcc, X

    for f in rttm_files.values():
        f.close()

    # --- agrégats micro ---
    agg = []
    for name in variants:
        df = pd.DataFrame(rows[name])
        if "p_chosen" in df and df["p_chosen"].isna().all():
            df = df.drop(columns="p_chosen")
        df.to_csv(OUT_DIR / f"{name}_{tag}_par_fichier.csv", index=False)
        m, tot = metrics[name], df["total_s"].sum()
        agg.append({
            "variante": name, "n_fichiers": len(df),
            "DER_%": 100 * abs(m["der"]), "DER_forgiving_%": 100 * abs(m["derf"]), "JER_%": 100 * abs(m["jer"]),
            "FA_%": 100 * df["FA_s"].sum() / tot, "missed_%": 100 * df["missed_s"].sum() / tot,
            "confusion_%": 100 * df["confusion_s"].sum() / tot,
            "VAD_error_%": 100 * abs(det),
            "RTF": (df["RTF"] * df["duration_s"]).sum() / df["duration_s"].sum(),
            "k_exact_%": 100 * (df["k_hyp"] == df["n_ref_speakers"]).mean(),
            "k_sous_estime_%": 100 * (df["k_hyp"] < df["n_ref_speakers"]).mean(),
            "k_sur_estime_%": 100 * (df["k_hyp"] > df["n_ref_speakers"]).mean(),
        })
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT_DIR / f"agregats_{tag}.csv", index=False)
    print(f"\n=== NME-SC, {tag}, VAD best_vad(train={VAD_TRAIN!r}), agrégation micro ===")
    print(agg.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
