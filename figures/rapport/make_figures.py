"""Figures du rapport final (docs/rapport-final.md), générées à partir de résultats existants.

Aucun pipeline n'est relancé et aucun fichier du test n'est lu par ce script :
- schema_chaine_baseline.png : schéma de la chaîne de baseline.py (valeurs recopiées de baseline.py) ;
- decomposition_oracle_dev90.png : FA / missed / confusion des diagnostics oracle du 11/11 (dev sample 90),
  recalculés avec pyannote.metrics depuis les RTTM d'hypothèses du commit 4ae8849 (lus par `git show`) ;
- decomposition_ablation_test90.png : FA / missed / confusion des 4 configurations de l'ablation, relus dans
  results/ablation/ablation_test_sample90_seed130_agregats.csv (run du 14/11, aucun recalcul) ;
- sample90_vs_dev.png : distributions par fichier du dev complet et du dev sample 90, d'après
  figures/voxconverse_file_stats.csv (lignes du dev uniquement) et samples/dev/sample90_seed130.rttm.

Protocole des recalculs (D-30 à D-37) : collar 0, UEM = [0, durée du wav], agrégation micro.
Lancement, depuis la racine du dépôt, .venv activé : python figures/rapport/make_figures.py
"""
import subprocess
from decimal import ROUND_HALF_UP, Decimal
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "rapport"
DEV_RTTM = ROOT / "samples" / "dev" / "sample90_seed130.rttm"
DEV_AUDIO = ROOT / "data" / "voxconverse" / "dev" / "audio"
ORACLE_COMMIT = "4ae8849"  # commit du 11/11 : runs de baseline_oracle.py utilisés dans les notes et le diagnostic k
def fr(v):
    """Arrondi à 2 décimales, au plus proche (demi vers le haut), virgule décimale."""
    return str(Decimal(f"{v:.4f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)).replace(".", ",")


COLORS = {"FA": "#E69F00", "missed": "#56B4E9", "confusion": "#CC79A7"}


# ---------------------------------------------------------------- évaluation
def evaluate(hyp_rttm, ref_rttm=DEV_RTTM, audio_dir=DEV_AUDIO):
    """DER moderne, DER forgiving, JER et décomposition (en % de la parole de référence), agrégation micro."""
    refs, hyps = load_rttm(str(ref_rttm)), load_rttm(str(hyp_rttm))
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_f = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0)
    for uri in sorted(refs):
        uem = Timeline([Segment(0, sf.info(str(audio_dir / f"{uri}.wav")).duration)])
        der(refs[uri], hyps[uri], uem=uem)
        der_f(refs[uri], hyps[uri], uem=uem)
        jer(refs[uri], hyps[uri], uem=uem)
    c = der[:]
    tot = c["total"]
    return {"DER": 100 * abs(der), "DER_forgiving": 100 * abs(der_f), "JER": 100 * abs(jer),
            "FA": 100 * c["false alarm"] / tot, "missed": 100 * c["missed detection"] / tot,
            "confusion": 100 * c["confusion"] / tot}


def oracle_table():
    runs = [("Baseline du 11/11\n(aucun oracle)", "baseline_oracle_vad0_k0_vote0"),
            ("VAD oracle", "baseline_oracle_vad1_k0_vote0"),
            ("vote oracle", "baseline_oracle_vad0_k0_vote1"),
            ("k oracle", "baseline_oracle_vad0_k1_vote0")]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, name in runs:
            path = Path(tmp) / f"{name}.rttm"
            blob = subprocess.run(["git", "-C", str(ROOT), "show",
                                   f"{ORACLE_COMMIT}:results/{name}_sample90_seed130.rttm"],
                                  check=True, capture_output=True).stdout
            path.write_bytes(blob)
            rows.append({"configuration": label.replace("\n", " "), "label": label, **evaluate(path)})
    return pd.DataFrame(rows)


def ablation_table():
    df = pd.read_csv(ROOT / "results" / "ablation" / "ablation_test_sample90_seed130_agregats.csv")
    labels = ["1. energy_vad\n+ p fixe 0.1\n(départ)", "2. best_vad\n+ p fixe 0.1\n(+ n°1)",
              "3. energy_vad\n+ NME-SC\n(+ n°2)", "4. best_vad\n+ NME-SC\n(les deux)"]
    return pd.DataFrame({"label": labels, "DER": df["DER_%"], "JER": df["JER_%"], "FA": df["FA_%"],
                         "missed": df["missed_%"], "confusion": df["confusion_%"]})


# ---------------------------------------------------------------- figures
def stacked_bars(df, title, path, note):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(df))
    bottom = [0.0] * len(df)
    for comp, name in [("FA", "false alarm"), ("missed", "missed"), ("confusion", "confusion")]:
        vals = df[comp].tolist()
        ax.bar(x, vals, bottom=bottom, color=COLORS[comp], label=name, width=0.6, edgecolor="white")
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v >= 1.5:
                ax.text(i, b + v / 2, fr(v), ha="center", va="center", fontsize=9)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, (tot, j) in enumerate(zip(df["DER"], df["JER"])):
        ax.text(i, tot + 0.6, f"DER {fr(tot)} %\nJER {fr(j)} %", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("% de la parole de référence (DER moderne)")
    ax.set_ylim(0, max(bottom) * 1.38)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, 0.01, note, fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def schema(path):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="0.25", lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.2, linespacing=1.35)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="0.2"))

    grey, blue, green, orange = "#EEEEEE", "#DCEBF7", "#E3F2DF", "#FCE9D2"
    # ligne du haut : signal
    box(0.1, 3.3, 1.9, 1.9, "1. Audio", "librosa\n16 kHz mono\n(D-50)", grey)
    box(2.4, 3.3, 2.2, 1.9, "2. STFT (frames)", "fenêtre de Hann\nframe 65 ms\nhop 32,5 ms", grey)
    box(5.0, 3.3, 1.9, 1.9, "4. MFCC", "13 coefficients\npar frame", grey)
    box(7.3, 3.3, 2.5, 1.9, "5. Embeddings\n(windows)", "\nwindows 2,0 s, hop 1,0 s,\nmin 0,5 s, dans la parole\nmoyenne + écart-type\ndes MFCC (26 dim)", grey)
    # VAD en dessous, branchée sur l'audio
    box(2.4, 0.35, 4.5, 2.35, "3. VAD", "amélioration n°1 : best_vad (vad_stack)\n2 étages HistGradientBoosting,\nmodèles devtrain (mesures dev) / alldev\n(test, démo)\navant : energy_vad, seuil −30 dB,\ndurées min. 0,5 s / 0,25 s", orange)
    box(7.3, 0.35, 2.5, 2.35, "6. Clustering\nspectral", "\ncosinus, graphe à p voisins\namélioration n°2 : p choisi\npar fichier (NME-SC)\nk par eigengap (1 à 20)\nKMeans seed 42", blue)
    box(10.2, 0.35, 2.7, 2.35, "7. Post-traitement", "\nvote des windows → frames\n(windows_to_annotation)\ndépartage par score\n3e amélioration (V3) :\nfill_unlabeled", green)
    box(10.2, 3.3, 2.7, 1.9, "Sortie", "Annotation pyannote\n→ RTTM\névaluation : DER, DER\nforgiving, JER, RTF", grey)

    arrow((2.05, 4.25), (2.35, 4.25))
    arrow((4.65, 4.25), (4.95, 4.25))
    arrow((6.95, 4.25), (7.25, 4.25))
    arrow((1.05, 3.25), (2.35, 1.55))          # audio -> VAD
    arrow((6.95, 2.2), (7.6, 3.25))            # VAD -> embeddings (segments de parole)
    arrow((8.55, 3.25), (8.55, 2.75))          # embeddings -> clustering
    arrow((9.85, 1.5), (10.15, 1.5))           # clustering -> post-traitement
    arrow((6.95, 0.9), (10.15, 0.75))          # VAD -> post-traitement (frames de parole)
    arrow((11.55, 2.75), (11.55, 3.25))        # post-traitement -> sortie
    ax.text(7.0, 2.85, "segments\nde parole", fontsize=7.5, color="0.3")
    ax.set_title("Chaîne de la baseline (baseline.py, diarize_file) — un fichier à la fois", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_vs_dev(path):
    stats = pd.read_csv(ROOT / "figures" / "voxconverse_file_stats.csv")
    dev = stats[stats["split"] == "dev"]
    uris = set(load_rttm(str(DEV_RTTM)))
    sample = dev[dev["file_id"].isin(uris)]
    assert len(dev) == 216 and len(sample) == 90, (len(dev), len(sample))
    metrics = [("n_speakers", "nombre de locuteurs"), ("overlap_ratio_pct", "overlap (% du temps de parole)"),
               ("speech_ratio_pct", "parole (% de la durée)"), ("audio_duration_min", "durée (min)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.3))
    for ax, (col, name) in zip(axes, metrics):
        data = [dev[col].values, sample[col].values]
        ax.boxplot(data, widths=0.5, showfliers=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"dev\n(n={len(dev)})", f"sample 90\n(n={len(sample)})"])
        ax.set_title(f"{name}\nmédianes : {pd.Series(data[0]).median():.2f} / {pd.Series(data[1]).median():.2f}"
                     .replace(".", ","), fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("VoxConverse v0.3 : dev complet (216 fichiers) et dev sample 90 (seed 130), un point par fichier",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schema(OUT / "schema_chaine_baseline.png")

    ora = oracle_table()
    ora.drop(columns="label").to_csv(OUT / "decomposition_oracle_dev90.csv", index=False, float_format="%.4f")
    print(ora.drop(columns="label").round(2).to_string(index=False))
    stacked_bars(ora, "Diagnostics oracle : un composant remplacé par la vérité terrain (dev sample 90)",
                 OUT / "decomposition_oracle_dev90.png",
                 "Runs de baseline_oracle.py du 11/11 (commit 4ae8849) : energy_vad, MFCC, windows 2 s / 1 s, "
                 "spectral p = 0.1,\nmax 10 locuteurs, KMeans seed 42. Collar 0, UEM = durée du wav, micro.")

    abl = ablation_table()
    print(abl.round(2).to_string(index=False))
    stacked_bars(abl, "Ablation des deux améliorations (test sample 90, run du 14/11)",
                 OUT / "decomposition_ablation_test90.png",
                 "Source : results/ablation/ablation_test_sample90_seed130_agregats.csv. best_vad alldev ;\n"
                 "frames, MFCC, windows, max 20 locuteurs et KMeans seed 42 identiques. Collar 0, micro.")

    sample_vs_dev(OUT / "sample90_vs_dev.png")
    print(f"figures écrites dans {OUT}")


if __name__ == "__main__":
    main()
