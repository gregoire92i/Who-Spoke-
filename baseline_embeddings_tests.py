"""Balayage de la taille des windows d'embedding (win_s) de la baseline classique.
15 valeurs de win_s réparties linéairement entre 0,5 et 3,0 s, hop_s = 50 % de win_s.
Tout le reste (frames, VAD, MFCC, stats, clustering, post-traitement, métriques) est identique à baseline.py :
les paramètres sont importés depuis baseline.py, ils ne sont jamais recopiés.

Optimisation : la partie amont (audio, STFT, VAD, MFCC) ne dépend pas de win_s.
Elle est donc calculée UNE fois par fichier, puis on boucle sur les 15 tailles de windows.
Les fichiers restent traités un par un (un seul audio / une seule STFT en RAM)."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")#pas d'affichage (plt.show ne marche pas sous WSL) : on sauvegarde la figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

import baseline as B#paramètres de la baseline (sr, frame_size, seuil_db, stats, oracle, collar, sample_rttm...)
from audio_features import load_audio, FFT, energy_vad, compute_mfcc
from embeddings import compute_embeddings
from clustering import spectral_clustering
from post_treatment import windows_to_annotation


#### PARAMÈTRES DU BALAYAGE ####
win_values = np.round(np.linspace(0.5, 3.0, 15), 3)#15 tailles de windows (s)
hop_ratio = 0.5#hop_s = 50 % de win_s
min_s = B.min_s#fixe pour toutes les configurations (0,5 s) : toujours <= win_s ici
out_dir = Path("results/sweep_windows")
fig_dir = Path("figures")
run_name = "sweep_windows_sample90_seed130"


def front_end(path, uri):
    """Partie amont, indépendante de win_s : audio -> STFT -> VAD -> MFCC."""
    audio = load_audio(str(path), B.sr)
    D = FFT(audio, frame_size=B.frame_size, hop_size=B.hop_size)
    del audio
    vad = energy_vad(D, sr=B.sr, frame_size=B.frame_size, hop_size=B.hop_size, seuil_db=B.seuil_db,
                     min_silence=B.min_silence, min_speech=B.min_speech, uri=uri)
    mfcc = compute_mfcc(D, sr=B.sr, n_mfcc=B.n_mfcc)
    del D
    return vad, mfcc


def back_end(mfcc, vad, uri, n_ref_speakers, *, win_s, hop_s):
    """Partie aval, qui dépend de win_s : embeddings -> clustering -> post-traitement."""
    X, win_segs = compute_embeddings(mfcc, vad, win_s=win_s, hop_s=hop_s, min_s=min_s, stats=B.stats,
                                     drop_c0=B.drop_c0, mode=B.mode, sr=B.sr, hop_size=B.hop_size)
    labels, info = spectral_clustering(X, oracle=B.oracle, n_speakers=n_ref_speakers, return_info=True)
    hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                      sr=B.sr, hop_size=B.hop_size, return_info=True)
    stats_file = {
        "k_hyp": info["k"],
        "n_windows": len(X),
        "n_speech_frames": post["n_speech"],
        "n_uncovered": post["n_uncovered"],
        "n_tie": post["n_tie"],
    }
    return hyp, stats_file


def main():
    #annotations du sample, restreintes aux fichiers dont le wav existe (même filtre que baseline.py)
    sample_annotations = {}
    for uri, ann in load_rttm(B.sample_rttm).items():
        if (B.audio_dir / f"{uri}.wav").exists():
            sample_annotations[uri] = ann
        else:
            print(f"[warning] {B.audio_dir / f'{uri}.wav'} introuvable, fichier ignoré")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs = [(float(w), round(float(w) * hop_ratio, 4)) for w in win_values]
    print(f"Configurations (win_s, hop_s) : {configs}")

    #une instance de chaque métrique PAR configuration : l'agrégation micro se fait config par config
    metrics = {w: {"der": DiarizationErrorRate(collar=B.collar, skip_overlap=False),
                   "der_forgiving": DiarizationErrorRate(collar=B.collar, skip_overlap=True),
                   "jer": JaccardErrorRate(collar=B.collar, skip_overlap=False)}
               for w, _ in configs}
    det = DetectionErrorRate(collar=B.collar, skip_overlap=False)#la VAD ne dépend pas de win_s : une seule fois
    rttm_files = {w: open(out_dir / f"{run_name}_win{w:.3f}.rttm", "w") for w, _ in configs}

    rows = []
    n_files = len(sample_annotations)
    try:
        for i, uri in enumerate(sorted(sample_annotations), start=1):
            ref = sample_annotations[uri]
            path = B.audio_dir / f"{uri}.wav"
            n_ref = len(ref.labels())
            duration = sf.info(str(path)).duration
            uem = Timeline([Segment(0, duration)], uri=uri)

            t0 = time.perf_counter()
            vad, mfcc = front_end(path, uri)
            t_front = time.perf_counter() - t0
            det(ref, vad.to_annotation(), uem=uem)

            msg = []
            for win_s, hop_s in configs:
                t0 = time.perf_counter()
                hyp, stats_file = back_end(mfcc, vad, uri, n_ref, win_s=win_s, hop_s=hop_s)
                t_back = time.perf_counter() - t0
                m = metrics[win_s]
                row = {
                    "win_s": win_s, "hop_s": hop_s, "uri": uri, "duration_s": duration,
                    "n_ref_speakers": n_ref, **stats_file,
                    "DER": m["der"](ref, hyp, uem=uem),
                    "DER_forgiving": m["der_forgiving"](ref, hyp, uem=uem),
                    "JER": m["jer"](ref, hyp, uem=uem),
                    #RTF comparable à baseline.py : amont (partagé) + aval de cette configuration
                    "RTF": (t_front + t_back) / duration,
                }
                rows.append(row)
                hyp.write_rttm(rttm_files[win_s])
                msg.append(f"{win_s:.2f}s:{row['DER']:.1%}")
            print(f"[{i}/{n_files}] {uri} | " + " ".join(msg))
            del mfcc, vad
    finally:
        for f in rttm_files.values():
            f.close()

    #résultats par fichier et par configuration
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}_par_fichier.csv", index=False)

    #tableau récapitulatif : une ligne par taille de window, agrégation micro
    summary = []
    for win_s, hop_s in configs:
        m, d = metrics[win_s], df[df["win_s"] == win_s]
        total = m["der"]["total"]#durée de parole de référence cumulée (s)
        n_speech = d["n_speech_frames"].sum()
        summary.append({
            "win_s": win_s, "hop_s": hop_s,
            "DER": abs(m["der"]),
            "DER_forgiving": abs(m["der_forgiving"]),
            "JER": abs(m["jer"]),
            "confusion": m["der"]["confusion"] / total,
            "false_alarm": m["der"]["false alarm"] / total,
            "missed": m["der"]["missed detection"] / total,
            "k_err_moyen": (d["k_hyp"] - d["n_ref_speakers"]).abs().mean(),#erreur moyenne sur le nb de locuteurs
            "n_windows": d["n_windows"].sum(),
            "uncovered": d["n_uncovered"].sum() / n_speech,
            "tie": d["n_tie"].sum() / n_speech,
            "RTF": (d["RTF"] * d["duration_s"]).sum() / d["duration_s"].sum(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / f"{run_name}_resume.csv", index=False)

    pct = ["DER", "DER_forgiving", "JER", "confusion", "false_alarm", "missed", "uncovered", "tie"]
    show = summary.copy()
    show[pct] = (show[pct] * 100).round(2)
    print(f"\n=== {run_name} : {n_files} fichiers, agrégation micro (valeurs en %) ===")
    print(show.to_string(index=False))
    print(f"\nVAD error (identique pour toutes les configurations) : {abs(det):.2%}")
    best = summary.loc[summary["DER"].idxmin()]
    print(f"Meilleur DER moderne : {best['DER']:.2%} pour win_s = {best['win_s']:.2f} s (hop {best['hop_s']:.2f} s)")

    #figure : DER en fonction de win_s, avec la décomposition
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["win_s"], summary["DER"] * 100, "o-", label="DER moderne")
    ax.plot(summary["win_s"], summary["DER_forgiving"] * 100, "s--", label="DER forgiving")
    ax.plot(summary["win_s"], summary["confusion"] * 100, "^:", label="dont confusion")
    ax.set_xlabel("taille des windows d'embedding win_s (s), hop = 50 %")
    ax.set_ylabel("%")
    ax.set_title(f"Baseline : DER en fonction de win_s ({n_files} fichiers, collar {B.collar})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{run_name}.png", dpi=150)
    print(f"Figure : {fig_dir / f'{run_name}.png'}")


if __name__ == "__main__":
    main()