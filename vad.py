"""VAD par énergie (D-52, D-53, D-54). Module réutilisable : aucun code exécuté à l'import. -> codé par claude pour les test de vad"""
import numpy as np
import librosa
from pyannote.core import Segment, Timeline

# paramètres (cf. D-50, D-51)
sr = 16000                       # fréquence d'échantillonnage
frame_size = int(64 * sr / 1000) # 64 ms -> 1024 échantillons
hop_size = int(10 * sr / 1000)   # 10 ms -> 160 échantillons


def load_audio(path, sr=sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def FFT(signal_audio, frame_size=frame_size, hop_size=hop_size):
    """STFT du signal ; tailles en nombre d'échantillons."""
    return librosa.stft(signal_audio, n_fft=frame_size, hop_length=hop_size, win_length=frame_size)


def RMS_on_FFT(FFT_output, frame_size=frame_size):
    """Énergie RMS par trame, calculée depuis la STFT. Forme (1, n_trames)."""
    return librosa.feature.rms(S=np.abs(FFT_output), frame_length=frame_size)


def compute_rms_db(D, frame_size=frame_size):
    """Énergie par trame en dB relatifs au max du fichier. Forme (n_trames,)."""
    return librosa.amplitude_to_db(RMS_on_FFT(D, frame_size=frame_size), ref=np.max)[0]


def vad_from_rms_db(rms_db, sr=sr, hop_size=hop_size,
                    seuil_db=-23.0, min_silence=0.25, min_speech=0.1, uri=None):
    """Seuil + smoothing par durées minimales. Renvoie une Timeline des zones de parole."""
    is_speech = (rms_db > seuil_db).astype(int)
    d = np.diff(is_speech, prepend=0, append=0)          # +1 = début, -1 = fin
    t_starts = librosa.frames_to_time(np.flatnonzero(d == 1), sr=sr, hop_length=hop_size)
    t_ends = librosa.frames_to_time(np.flatnonzero(d == -1), sr=sr, hop_length=hop_size)
    tl = Timeline([Segment(s, e) for s, e in zip(t_starts, t_ends)], uri=uri)
    tl = tl.support(collar=min_silence)                   # 1) combler les trous courts
    return Timeline([seg for seg in tl if seg.duration >= min_speech], uri=uri)  # 2) puis retirer la parole courte


def energy_vad(D, sr=sr, frame_size=frame_size, hop_size=hop_size,
               seuil_db=-23.0, min_silence=0.25, min_speech=0.1, uri=None):
    """VAD complète à partir de la STFT D (sortie de FFT)."""
    rms_db = compute_rms_db(D, frame_size=frame_size)
    return vad_from_rms_db(rms_db, sr=sr, hop_size=hop_size, seuil_db=seuil_db,
                           min_silence=min_silence, min_speech=min_speech, uri=uri)


"""Évaluation de la VAD par énergie (energy_vad) sur un sample du dev VoxConverse.
Protocole : DetectionErrorRate, collar 0 (D-31), UEM [0, durée wav] (D-33), agrégation micro (D-34).
Sorties : figures/vad_eval.csv (toutes les combinaisons) et figures/vad_eval.png (tableau des meilleures)."""
import itertools
import time
from pathlib import Path
 
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
 
from vad import load_audio, FFT, energy_vad
from rttm_files import get_representative_sample
 
AUDIO_DIR = Path("data/voxconverse/dev/audio")
RTTM_DIR = Path("data/voxconverse/annotations/dev")
N_FILES, SEED = 40, 130
TOP = 15                                       # nombre de lignes dans le tableau PNG
 #METRIQUES
SEUILS_DB = list(range(-50, -9, 2))            # dB sous le max du fichier
MIN_SILENCES = [0.0, 0.1, 0.25, 0.5]           # s
MIN_SPEECHES = [0.0, 0.05, 0.1, 0.2, 0.3]      # s
 
 
def main():
    uris = [Path(f).stem for f in get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", N_FILES,seed=SEED)]
    combos = list(itertools.product(SEUILS_DB, MIN_SILENCES, MIN_SPEECHES))
    metrics = {c: DetectionErrorRate(collar=0.0, skip_overlap=False) for c in combos}
    metrics["tout parole"] = DetectionErrorRate(collar=0.0, skip_overlap=False)   # VAD triviale
 
    t0 = time.perf_counter()
    for i, uri in enumerate(uris, 1):
        wav = AUDIO_DIR / f"{uri}.wav"
        ref = load_rttm(RTTM_DIR / f"{uri}.rttm")[uri]
        duree = sf.info(wav).duration
        uem = Timeline([Segment(0, duree)])
        D = FFT(load_audio(wav))                                   # STFT une fois par fichier
        for seuil, min_sil, min_sp in combos:
            hyp = energy_vad(D, seuil_db=seuil, min_silence=min_sil, min_speech=min_sp, uri=uri)
            metrics[(seuil, min_sil, min_sp)](ref, hyp.to_annotation(), uem=uem)
        metrics["tout parole"](ref, uem.to_annotation(), uem=uem)
        print(f"[{i}/{len(uris)}] {uri} ({duree:.0f} s) — {time.perf_counter() - t0:.0f} s écoulées")
 
    rows = []
    for c, m in metrics.items():
        acc = m.accumulated_
        seuil, min_sil, min_sp = c if c != "tout parole" else ("tout parole", "-", "-")
        rows.append({"seuil_db": seuil, "min_silence_s": min_sil, "min_speech_s": min_sp,
                     "parole_reconnue_%": 100 * (1 - acc["miss"] / acc["total"]),
                     "fausses_alarmes_%": 100 * acc["false alarm"] / acc["total"],
                     "detection_error_%": 100 * abs(m)})
    df = pd.DataFrame(rows).sort_values("detection_error_%").reset_index(drop=True)
 
    Path("figures").mkdir(exist_ok=True)
    df.to_csv("figures/vad_eval.csv", index=False, float_format="%.2f")
 
    # tableau PNG : les TOP meilleures combinaisons + la VAD triviale en dernière ligne
    show = pd.concat([df[df["seuil_db"] != "tout parole"].head(TOP),
                      df[df["seuil_db"] == "tout parole"]])
    cells = [[f"{v:.2f}" if isinstance(v, float) else str(v) for v in r] for r in show.values]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(cells) + 1.2))
    ax.axis("off")
    ax.table(cellText=cells, colLabels=list(show.columns), loc="center", cellLoc="center")
    ax.set_title(f"VAD énergie — {len(uris)} fichiers dev (seed {SEED}), collar 0, micro, tri par detection error")
    plt.savefig("figures/vad_eval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(df.head(TOP).to_string(index=False))
 
 
if __name__ == "__main__":
    main()