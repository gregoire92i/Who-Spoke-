"""Balayage frame_ms x hop_ms de la STFT : impact sur le DER final de la baseline.

Ce qui CHANGE   : frame_ms (taille de trame STFT, donc VAD + MFCC) et hop_pct (pas entre trames, en % de la frame).
Ce qui RESTE    : tout le reste, importé tel quel depuis baseline.py (seuil VAD, smoothing, n_mfcc,
                  fenêtres d'embedding, clustering, oracle, collar, sample, UEM, agrégation micro).

Sorties (dans results/sweep_frames_pct/) :
  - resume.csv      : 1 ligne par configuration (métriques micro sur le sample)
  - par_fichier.csv : 1 ligne par (configuration, fichier)
  - rttm/frame{F}_hop{P}pct.rttm : hypothèses, pour réévaluer sans relancer
  - figures/sweep_frames_DER.png

Reprise : une configuration déjà présente dans resume.csv n'est pas recalculée (on peut couper et relancer).
Usage :
  python sweep_frames.py --dry-run     # affiche la grille et le nombre de runs, ne calcule rien
  python sweep_frames.py               # lance (ou reprend) le balayage puis trace la figure
  python sweep_frames.py --plot-only   # retrace la figure depuis resume.csv
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")#plt.show ne marche pas sous WSL : on sauvegarde la figure
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

import baseline as B#paramètres + diarize_file : une seule source de vérité


#### GRILLE ####
#5 frames réparties de manière égale entre 15 et 70 ms (pas de 13,75 ms)
FRAMES_MS = [15 + i * (70 - 15) / 4 for i in range(5)]#15, 28.75, 42.5, 56.25, 70
#hop en % de la frame (la baseline actuelle 60/10 ms correspond à ~17 %)
HOPS_PCT = [10, 25, 50]

#à 16 kHz, 1 ms = 16 échantillons : on vérifie que chaque frame tombe sur un nombre entier d'échantillons
assert all(float(f * B.sr / 1000).is_integer() for f in FRAMES_MS)

out_dir = B.out_dir / "sweep_frames_pct"
fig_dir = Path("figures")
resume_csv = out_dir / "resume.csv"
par_fichier_csv = out_dir / "par_fichier.csv"


def grille():
    """Couples (frame_ms, hop_pct). Hop <= 50 % : les trames se chevauchent toujours."""
    return [(f, p) for f in FRAMES_MS for p in HOPS_PCT]


def run_config(frame_ms, hop_pct, annotations):
    """Toute la baseline sur le sample, pour UNE configuration. Métriques neuves à chaque appel :
    sinon pyannote accumulerait les erreurs de toutes les configurations."""
    frame_size = int(frame_ms * B.sr / 1000)
    hop_size = round(frame_size * hop_pct / 100)#arrondi à l'échantillon près
    hop_ms = hop_size * 1000 / B.sr#hop réellement utilisé, en ms

    der = DiarizationErrorRate(collar=B.collar, skip_overlap=False)
    der_forgiving = DiarizationErrorRate(collar=B.collar, skip_overlap=True)
    jer = JaccardErrorRate(collar=B.collar, skip_overlap=False)
    det = DetectionErrorRate(collar=B.collar, skip_overlap=False)

    rows, n_warn = [], 0
    with open(out_dir / "rttm" / f"frame{frame_ms}_hop{hop_pct}pct.rttm", "w") as f_rttm:
        for uri in sorted(annotations):#même ordre que la baseline
            ref = annotations[uri]
            path = B.audio_dir / f"{uri}.wav"
            duration = sf.info(str(path)).duration
            uem = Timeline([Segment(0, duration)], uri=uri)

            #on compte les warnings (ex. filtres mel vides si la trame est trop courte) au lieu de les ignorer
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                hyp, vad, stats_file = B.diarize_file(path, uri, n_ref_speakers=len(ref.labels()),
                                                      frame_size=frame_size, hop_size=hop_size)
                proc_time = time.perf_counter() - t0
            n_warn += len(w)

            rows.append({
                "frame_ms": frame_ms, "hop_pct": hop_pct, "hop_ms": hop_ms, "uri": uri,
                "duration_s": duration, "n_ref_speakers": len(ref.labels()), **stats_file,
                "DER": der(ref, hyp, uem=uem),
                "DER_forgiving": der_forgiving(ref, hyp, uem=uem),
                "JER": jer(ref, hyp, uem=uem),
                "VAD_error": det(ref, vad.to_annotation(), uem=uem),
                "RTF": proc_time / duration,
            })
            hyp.write_rttm(f_rttm)

    df = pd.DataFrame(rows)
    resume = {
        "frame_ms": frame_ms, "hop_pct": hop_pct, "hop_ms": hop_ms,
        "DER": abs(der), "DER_forgiving": abs(der_forgiving), "JER": abs(jer), "VAD_error": abs(det),
        "RTF": (df["RTF"] * df["duration_s"]).sum() / df["duration_s"].sum(),
        "k_abs_error_mean": (df["k_hyp"] - df["n_ref_speakers"]).abs().mean(),
        "uncovered": df["n_uncovered"].sum() / df["n_speech_frames"].sum(),
        "n_files": len(df), "n_warnings": n_warn,
    }
    return resume, df


def append_csv(df, path):
    """Écriture incrémentale : un crash au run 80 ne fait pas perdre les 79 premiers."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def plot(resume_path=resume_csv):
    res = pd.read_csv(resume_path).sort_values(["hop_pct", "frame_ms"])
    #couleurs catégorielles en ordre fixe + marqueurs différents : le hop n'est jamais codé par la couleur seule
    couleurs = ["#2a78d6", "#eb6834", "#1baf7a"]
    marqueurs = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for i, pct in enumerate(HOPS_PCT):
        d = res[res["hop_pct"] == pct]
        if d.empty:
            continue
        ax.plot(d["frame_ms"], d["DER"] * 100, color=couleurs[i], marker=marqueurs[i],
                markersize=6, linewidth=2, label=f"hop {pct} % de la frame")

    best = res.loc[res["DER"].idxmin()]
    ax.scatter(best["frame_ms"], best["DER"] * 100, s=160, facecolors="none", edgecolors="#1a1a19",
               linewidths=1.5, zorder=5,
               label=f"minimum : {best['DER']:.2%}\n(frame {best['frame_ms']:g} ms, "
                     f"hop {best['hop_pct']:.0f} % = {best['hop_ms']:g} ms)")

    ax.set_xlabel("taille de frame (ms)")
    ax.set_ylabel("DER moderne (%)  — collar 0, overlap compté")
    ax.set_title(f"DER moderne en fonction de la frame et du hop (% de la frame) — {Path(B.sample_rttm).name}, "
                 f"{int(res['n_files'].iloc[0])} fichiers, agrégation micro", fontsize=11)
    ax.set_xticks(FRAMES_MS, [f"{f:g}" for f in FRAMES_MS])
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.06, 1))
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "sweep_frames_DER.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Figure : {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    configs = grille()
    print(f"{len(FRAMES_MS)} frames x {len(HOPS_PCT)} hops -> {len(configs)} configurations")
    for f in FRAMES_MS:
        n = int(f * B.sr / 1000)
        print(f"frame {f:g} ms ({n} éch.) -> hops : "
              + ", ".join(f"{p} % = {round(n * p / 100) * 1000 / B.sr:g} ms" for p in HOPS_PCT))
    if args.dry_run:
        return
    if args.plot_only:
        plot()
        return

    #même filtrage que la baseline : uniquement les fichiers dont le wav existe
    annotations = {uri: ann for uri, ann in load_rttm(B.sample_rttm).items()
                   if (B.audio_dir / f"{uri}.wav").exists()}
    print(f"sample : {B.sample_rttm} ({len(annotations)} fichiers)")
    (out_dir / "rttm").mkdir(parents=True, exist_ok=True)

    deja_faits = set()
    if resume_csv.exists():
        prev = pd.read_csv(resume_csv)
        deja_faits = set(zip(prev["frame_ms"], prev["hop_pct"]))
        print(f"reprise : {len(deja_faits)} configurations déjà calculées")

    a_faire = [c for c in configs if c not in deja_faits]
    t_start = time.perf_counter()
    for i, (frame_ms, hop_pct) in enumerate(a_faire, start=1):
        t0 = time.perf_counter()
        resume, df = run_config(frame_ms, hop_pct, annotations)
        append_csv(pd.DataFrame([resume]), resume_csv)
        append_csv(df, par_fichier_csv)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (len(a_faire) - i)
        print(f"[{i}/{len(a_faire)}] frame {frame_ms:g} ms | hop {hop_pct} % ({resume['hop_ms']:g} ms) | DER {resume['DER']:.2%} "
              f"| forgiving {resume['DER_forgiving']:.2%} | warnings {resume['n_warnings']} "
              f"| {time.perf_counter() - t0:.0f} s | reste ~{eta / 60:.0f} min")

    res = pd.read_csv(resume_csv)
    print("\n=== 5 meilleures configurations (DER moderne) ===")
    print(res.nsmallest(5, "DER")[["frame_ms", "hop_pct", "hop_ms", "DER", "DER_forgiving", "JER", "VAD_error",
                                   "k_abs_error_mean", "RTF"]].to_string(index=False))
    if res["n_warnings"].sum() > 0:
        print("\n[attention] des warnings ont été levés : voir la colonne n_warnings de resume.csv")
    plot()


if __name__ == "__main__":
    main()