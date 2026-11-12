"""Test de la VAD par énergie de la baseline (energy_vad) : impact du seuil OU du delta d'hystérésis.
Protocole : DetectionErrorRate, collar 0 (D-31), UEM [0, durée wav] (D-33), agrégation micro (D-34).
detection error = (FA + MISS) / durée de parole de référence.
Sorties : figures/vad_<MODE>.csv et figures/vad_<MODE>.png"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")                       # pas d'affichage : la figure est sauvegardée dans figures/
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate


# ============================== MODE ==============================
MODE = "delta"            # "seuil" : on fait varier le seuil (delta = DELTA_FIXE)
                          # "delta" : on fait varier le delta (seuil = SEUIL_FIXE)
SEUIL_FIXE = -29.0        # dB, centre de l'hystérésis en mode "delta"
DELTA_FIXE = 0.0          # dB, delta utilisé en mode "seuil" (0 = pas d'hystérésis)
N_VALEURS = 50
SEUILS_DB = np.round(np.linspace(-40, -20, N_VALEURS), 2)   # mode "seuil" : 50 valeurs entre -40 et -20 dB
DELTAS_DB = np.round(np.linspace(0, 12, N_VALEURS), 2)      # mode "delta" : 50 valeurs entre 0 et 12 dB
# ==================================================================

# paramètres de la VAD : identiques à baseline.py
sr = 16000
frame_size = int(65 * sr / 1000)     # 65 ms   -> 1040 échantillons
hop_size = int(32.5 * sr / 1000)     # 32,5 ms -> 520 échantillons
SMOOTH_FRAMES = 3                    # moyenne glissante sur 3 frames
MIN_SILENCE, MIN_SPEECH = 0.5, 0.3   # s

# données
AUDIO_DIR = Path("data/voxconverse/dev/audio")
RTTM_DIR = Path("data/voxconverse/annotations/dev")
SAMPLE_RTTM = Path("samples/dev/sample90_seed130.rttm")   # sample de 90 fichiers du dev (même que la baseline)

# couleurs du graphe (une couleur par courbe, même ordre dans les deux modes)
BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#1baf7a"
ENCRE, ENCRE_2, GRILLE, AXE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ---------------- fonctions audio (mêmes que dans la baseline) ----------------
def load_audio(path, sr=sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def FFT(signal_audio, frame_size=frame_size, hop_size=hop_size):
    """STFT du signal ; tailles en nombre d'échantillons."""
    return librosa.stft(signal_audio, n_fft=frame_size, hop_length=hop_size, win_length=frame_size)


def RMS_on_FFT(FFT_output, frame_size=frame_size):
    """Énergie RMS par trame, calculée depuis la STFT. Forme (1, n_trames)."""
    return librosa.feature.rms(S=np.abs(FFT_output), frame_length=frame_size)


# ---------------- VAD : ta fonction, modifiée ----------------
def hysteresis(x, seuil_haut, seuil_bas):
    """Décision 0/1 par trame : on entre en parole si x > seuil_haut, on en sort si x <= seuil_bas.
    Avec seuil_haut == seuil_bas : identique à (x > seuil).astype(int)."""
    is_speech = np.zeros(len(x), dtype=int)
    etat = 0
    for i, v in enumerate(x):
        if etat == 0 and v > seuil_haut:
            etat = 1
        elif etat == 1 and v <= seuil_bas:
            etat = 0
        is_speech[i] = etat
    return is_speech


def energy_vad(D, sr=sr, frame_size=frame_size, hop_size=hop_size,
               seuil_db=-29.0, delta_db=0.0, smooth_frames=3,
               min_silence=0.5, min_speech=0.3, uri=None):
    """VAD par énergie : smoothing de l'énergie, hystérésis, puis durées minimales (D-53, D-54).
    D : sortie de FFT() (STFT complexe). Renvoie une Timeline des zones de parole.
    Hystérésis centrée sur seuil_db : entrée si > seuil_db + delta_db/2, sortie si <= seuil_db - delta_db/2.
    delta_db=0 et smooth_frames=1 -> ancienne VAD à seuil simple."""
    # 1. énergie par trame, en dB relatifs au max du fichier
    rms = RMS_on_FFT(D, frame_size=frame_size)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)[0]   # (1, n) -> (n,)

    # 1b. smoothing : moyenne glissante centrée sur smooth_frames trames
    #     mode="nearest" : on répète la valeur de bord (un padding à 0 dB = le max fausserait les bords)
    if smooth_frames > 1:
        rms_db = uniform_filter1d(rms_db, size=smooth_frames, mode="nearest")

    # 2. décision trame par trame (0/1) avec hystérésis
    is_speech = hysteresis(rms_db, seuil_haut=seuil_db + delta_db / 2, seuil_bas=seuil_db - delta_db / 2)

    # 3. changements d'état : +1 = début de parole, -1 = fin
    #    prepend/append=0 gèrent un fichier qui commence ou finit en parole
    d = np.diff(is_speech, prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)

    # 4. indices de trame -> secondes (sr explicite : sr de base librosa = 22050)
    t_starts = librosa.frames_to_time(starts, sr=sr, hop_length=hop_size)
    t_ends = librosa.frames_to_time(ends, sr=sr, hop_length=hop_size)
    tl = Timeline([Segment(s, e) for s, e in zip(t_starts, t_ends)], uri=uri)

    # 5. smoothing par durées : d'abord combler les trous courts, PUIS supprimer la parole courte
    tl = tl.support(collar=min_silence)
    tl = Timeline([seg for seg in tl if seg.duration >= min_speech], uri=uri)
    return tl


# ---------------- graphe ----------------
def trace(df, nom, n_files):
    x = df[nom].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if MODE == "delta":
        series = [("detection_error_%", "Detection error", BLEU),
                  ("FA_%", "FA (fausses alarmes)", ORANGE),
                  ("MISS_%", "MISS (parole manquée)", VERT)]
        ax.set_xlabel("delta d'hystérésis (dB)", color=ENCRE_2)
        titre = f"Impact du delta sur la detection error — seuil centré sur {SEUIL_FIXE:g} dB"
    else:
        series = [("detection_error_%", "Detection error", BLEU)]
        ax.set_xlabel("seuil (dB sous le max du fichier)", color=ENCRE_2)
        titre = f"Impact du seuil sur la detection error — delta {DELTA_FIXE:g} dB"

    for col, label, couleur in series:
        ax.plot(x, df[col], color=couleur, lw=2, marker="o", ms=3.5, label=label)

    # meilleur point (detection error minimale)
    i = df["detection_error_%"].idxmin()
    y_min = df.loc[i, "detection_error_%"]
    ax.plot(x[i], y_min, marker="o", ms=12, mfc="none", mec=ENCRE, mew=1.5)
    ax.annotate(f"min {y_min:.2f} % ({nom} = {x[i]:g})", (x[i], y_min), xytext=(0, 12),
                textcoords="offset points", ha="center", color=ENCRE, fontsize=9)

    ax.set_ylabel("% de la durée de parole de référence", color=ENCRE_2)
    ax.set_title(f"{titre}\n{n_files} fichiers dev ({SAMPLE_RTTM.name}), smoothing {SMOOTH_FRAMES} frames, "
                 f"min_silence {MIN_SILENCE} s, min_speech {MIN_SPEECH} s, collar 0, micro",
                 loc="left", fontsize=10, color=ENCRE)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRILLE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXE)
    ax.tick_params(colors=ENCRE_2)
    if len(series) > 1:
        ax.legend(frameon=False)

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/vad_{MODE}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- évaluation ----------------
def main():
    if MODE == "seuil":
        valeurs, nom = SEUILS_DB, "seuil_db"
        params = lambda v: dict(seuil_db=v, delta_db=DELTA_FIXE)
    elif MODE == "delta":
        valeurs, nom = DELTAS_DB, "delta_db"
        params = lambda v: dict(seuil_db=SEUIL_FIXE, delta_db=v)
    else:
        raise ValueError(f"MODE inconnu : {MODE!r} (attendu 'seuil' ou 'delta')")

    uris = sorted(load_rttm(SAMPLE_RTTM).keys())      # liste des fichiers du sample (90 uris)
    print(f"{len(uris)} fichiers lus dans {SAMPLE_RTTM}")
    metrics = {v: DetectionErrorRate(collar=0.0, skip_overlap=False) for v in valeurs}

    t0 = time.perf_counter()
    for i, uri in enumerate(uris, 1):
        wav = AUDIO_DIR / f"{uri}.wav"
        ref = load_rttm(RTTM_DIR / f"{uri}.rttm")[uri]
        duree = sf.info(wav).duration
        uem = Timeline([Segment(0, duree)])
        D = FFT(load_audio(wav))                                   # STFT une fois par fichier
        for v in valeurs:
            hyp = energy_vad(D, smooth_frames=SMOOTH_FRAMES, min_silence=MIN_SILENCE,
                             min_speech=MIN_SPEECH, uri=uri, **params(v))
            metrics[v](ref, hyp.to_annotation(), uem=uem)
        print(f"[{i}/{len(uris)}] {uri} ({duree:.0f} s) — {time.perf_counter() - t0:.0f} s écoulées")

    # detection error = (FA + MISS) / total, total = durée de parole de référence (agrégation micro)
    rows = []
    for v, m in metrics.items():
        acc = m.accumulated_
        rows.append({nom: v,
                     "detection_error_%": 100 * abs(m),
                     "FA_%": 100 * acc["false alarm"] / acc["total"],
                     "MISS_%": 100 * acc["miss"] / acc["total"]})
    df = pd.DataFrame(rows)

    Path("figures").mkdir(exist_ok=True)
    df.to_csv(f"figures/vad_{MODE}.csv", index=False, float_format="%.2f")
    trace(df, nom, len(uris))
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()