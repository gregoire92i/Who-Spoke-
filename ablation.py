"""Ablation des 2 améliorations de la baseline (D-62, protocole fixé le 14/11/2026 AVANT exécution).

4 configurations, chacune ne différant de sa voisine que par une amélioration :
                         | p fixe 0.1          | p par fichier (NME-SC)
  energy_vad de base     | baseline de départ  | + amélioration n°2
  best_vad (alldev)      | + amélioration n°1  | + les deux

Amélioration n°1 (VAD) : energy_vad -30 dB sans hystérésis ni lissage -> best_vad(train="alldev").
Amélioration n°2 (clustering) : spectral_clustering p_neighbors fixe 0.1 -> nme_spectral_clustering (Laplacien normalisé).
Tout le reste est identique et fixé EN DUR ci-dessous (indépendant de baseline.py, pour figer la version évaluée).

Fichiers : TEST sample 90 (mesure finale ; ne sert jamais à régler). Protocole D-30 à D-37 : collar 0,
UEM [0, durée du wav], micro, DER moderne / forgiving / JER, FA / missed / confusion, detection error, RTF.
Un fichier audio à la fois.

Usage : python ablation.py [--limit N]"""
import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from audio_features import load_audio, FFT, energy_vad, compute_mfcc
from vad import best_vad
from embeddings import compute_embeddings
from clustering import spectral_clustering, nme_spectral_clustering
from post_treatment import windows_to_annotation

#### PARAMÈTRES FIGÉS (D-62, 14/11/2026) ####
SAMPLE_RTTM = "samples/test/sample90_seed130.rttm"
AUDIO_DIR = Path("data/voxconverse/test/voxconverse_test_wav")  # D-12 : jamais __MACOSX
OUT_DIR = Path("results/ablation")
RUN_NAME = "ablation_test_sample90_seed130"

SR = 16000
FRAME_MS, HOP_MS = 65, 32.5
FRAME_SIZE = int(FRAME_MS * SR / 1000)
HOP_SIZE = int(HOP_MS * SR / 1000)
N_MFCC = 13
# VAD avant amélioration : energy_vad de base
ENERGY = dict(seuil_db=-30.0, delta_db=0.0, smooth_frames=1, min_silence=0.5, min_speech=0.25)
# VAD après amélioration
BEST_VAD_TRAIN = "alldev"  # autorisé sur le test : modèles entraînés sur le dev uniquement (D-57)
# embeddings (identiques partout)
EMB = dict(win_s=2.0, hop_s=1.0, min_s=0.5, stats="mean_std", drop_c0=False, mode="segment")
# clustering (communs)
CLUST = dict(oracle=False, min_speakers=1, max_speakers=20, standardize=True, seed=42, return_info=True)
P_FIXE, MIN_NEIGHBORS = 0.1, 5
NME = dict(min_neighbors=MIN_NEIGHBORS, max_frac=0.25, n_grid=30, laplacian="normalized")
COLLAR = 0.0

VADS = ("energy", "best")
CLUSTERINGS = ("fixe", "nme")
CONFIG_NAMES = {("energy", "fixe"): "1_energy_pfixe (baseline de départ)",
                ("best", "fixe"): "2_best_pfixe (+ amélioration n°1)",
                ("energy", "nme"): "3_energy_nme (+ amélioration n°2)",
                ("best", "nme"): "4_best_nme (+ les deux)"}


def cluster(X, method):
    if method == "fixe":
        return spectral_clustering(X, p_neighbors=P_FIXE, min_neighbors=MIN_NEIGHBORS, **CLUST)
    return nme_spectral_clustering(X, **NME, **CLUST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="n premiers fichiers (test rapide)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = RUN_NAME + (f"_limit{args.limit}" if args.limit else "")

    annotations = {}
    for uri, ann in load_rttm(SAMPLE_RTTM).items():
        if (AUDIO_DIR / f"{uri}.wav").exists():
            annotations[uri] = ann
        else:
            print(f"[warning] {uri}.wav introuvable, fichier ignoré")
    uris = sorted(annotations)[:args.limit] if args.limit else sorted(annotations)

    configs = [(v, c) for c in CLUSTERINGS for v in VADS]
    m = {cfg: dict(der=DiarizationErrorRate(collar=COLLAR, skip_overlap=False),
                   derf=DiarizationErrorRate(collar=COLLAR, skip_overlap=True),
                   jer=JaccardErrorRate(collar=COLLAR, skip_overlap=False)) for cfg in configs}
    det = {v: DetectionErrorRate(collar=COLLAR, skip_overlap=False) for v in VADS}
    rows = {cfg: [] for cfg in configs}
    rttm = {cfg: open(OUT_DIR / f"{run}_{v}_{c}.rttm", "w") for cfg in configs for v, c in [cfg]}

    for i, uri in enumerate(uris, start=1):
        ref = annotations[uri]
        path = AUDIO_DIR / f"{uri}.wav"
        duration = sf.info(str(path)).duration  # D-15
        uem = Timeline([Segment(0, duration)], uri=uri)  # D-33

        t0 = time.perf_counter()
        audio = load_audio(str(path), SR)
        D = FFT(audio, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
        mfcc = compute_mfcc(D, sr=SR, n_mfcc=N_MFCC)
        t_common = time.perf_counter() - t0

        line = f"[{i}/{len(uris)}] {uri} (réf {len(ref.labels())})"
        for v in VADS:
            t1 = time.perf_counter()
            if v == "energy":
                vad = energy_vad(D, sr=SR, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, uri=uri, **ENERGY)
            else:
                vad = best_vad(audio, uri=uri, train=BEST_VAD_TRAIN)
            X, win_segs = compute_embeddings(mfcc, vad, sr=SR, hop_size=HOP_SIZE, **EMB)
            t_vad_emb = time.perf_counter() - t1
            vad_err = det[v](ref, vad.to_annotation(), uem=uem)
            for c in CLUSTERINGS:
                cfg = (v, c)
                t2 = time.perf_counter()
                labels, info = cluster(X, c)
                hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                                  sr=SR, hop_size=HOP_SIZE, return_info=True)
                t_clust = time.perf_counter() - t2
                if hyp is None:
                    hyp = Annotation(uri=uri)
                comp = m[cfg]["der"](ref, hyp, uem=uem, detailed=True)
                rows[cfg].append({
                    "uri": uri, "duration_s": duration, "n_ref_speakers": len(ref.labels()),
                    "k_hyp": info["k"], "p_neighbors_used": info.get("p"), "n_windows": len(X),
                    "DER": comp["diarization error rate"],
                    "DER_forgiving": m[cfg]["derf"](ref, hyp, uem=uem),
                    "JER": m[cfg]["jer"](ref, hyp, uem=uem),
                    "VAD_error": vad_err,
                    "FA_s": comp["false alarm"], "missed_s": comp["missed detection"],
                    "confusion_s": comp["confusion"], "total_s": comp["total"],
                    "n_speech_frames": post["n_speech"], "n_uncovered": post["n_uncovered"], "n_tie": post["n_tie"],
                    # RTF : partie commune (chargement, STFT, MFCC) + VAD et embeddings + clustering et post-traitement
                    "RTF": (t_common + t_vad_emb + t_clust) / duration,
                })
                hyp.write_rttm(rttm[cfg])
                line += f" | {v}/{c}: {rows[cfg][-1]['DER']:.1%} k{info['k']}"
        print(line, flush=True)
        del audio, D, mfcc

    for f in rttm.values():
        f.close()

    agg = []
    for cfg in sorted(configs, key=lambda x: CONFIG_NAMES[x]):
        v, c = cfg
        df = pd.DataFrame(rows[cfg])
        df.to_csv(OUT_DIR / f"{run}_{v}_{c}_par_fichier.csv", index=False)
        tot = df["total_s"].sum()
        agg.append({
            "configuration": CONFIG_NAMES[cfg], "vad": v, "clustering": c, "n_fichiers": len(df),
            "DER_%": 100 * abs(m[cfg]["der"]), "DER_forgiving_%": 100 * abs(m[cfg]["derf"]),
            "JER_%": 100 * abs(m[cfg]["jer"]),
            "FA_%": 100 * df["FA_s"].sum() / tot, "missed_%": 100 * df["missed_s"].sum() / tot,
            "confusion_%": 100 * df["confusion_s"].sum() / tot,
            "VAD_detection_error_%": 100 * abs(det[v]),
            "RTF": (df["RTF"] * df["duration_s"]).sum() / df["duration_s"].sum(),
            "k_exact_%": 100 * (df["k_hyp"] == df["n_ref_speakers"]).mean(),
            "k_sous_estime_%": 100 * (df["k_hyp"] < df["n_ref_speakers"]).mean(),
            "k_sur_estime_%": 100 * (df["k_hyp"] > df["n_ref_speakers"]).mean(),
        })
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT_DIR / f"{run}_agregats.csv", index=False)
    print(f"\n=== Ablation {run} : agrégation micro, collar 0 ===")
    print(agg.drop(columns=["vad", "clustering"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
