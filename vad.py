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




# =====================================================================================
#                    VAD AGENT : VAD retenue (vad_stack) + banc d'évaluation
# Sorties dans results/vad_agent/. La référence est energy_vad d'audio_features.py (importée, jamais modifiée),
# avec les paramètres de baseline.py. Évaluation identique à main() : DetectionErrorRate collar 0,
# UEM [0, durée wav], agrégation micro. Historique complet des essais : results/vad_agent/journal.md.
# =====================================================================================
import argparse
from multiprocessing import Pool

AGENT_DIR = Path("results/vad_agent")
CACHE_DIR = AGENT_DIR / "cache"
SPLITS = {"dev": (Path("data/voxconverse/dev/audio"), Path("data/voxconverse/annotations/dev")),
          "test": (Path("data/voxconverse/test/voxconverse_test_wav"), Path("data/voxconverse/annotations/test"))}
# paramètres de la VAD de référence = ceux de baseline.py (delta_db, smooth_frames : défauts d'audio_features)
REF_PARAMS = dict(seuil_db=-30.0, min_silence=0.5, min_speech=0.25)
TUNE_RTTM = "samples/dev/sample90_seed130.rttm"
ALL_DEV_RTTM = "samples/dev/all_dev.rttm"


def import_energy_vad_ref():
    """energy_vad de référence (audio_features.py, non modifiée)."""
    from audio_features import energy_vad
    return energy_vad


# ---------------- caractéristiques trame par trame (même trame/hop que la baseline) ----------------
def periodicity(y, sr=sr, frame_size=frame_size, hop_size=hop_size, fmin=60.0, fmax=400.0, chunk=1500):
    """Harmonicité : max de l'autocorrélation normalisée sur les retards [sr/fmax, sr/fmin],
    corrigée de l'effet de la fenêtre de Hann. Renvoie (valeur max, f0 correspondante)."""
    n_fft = 2048                                   # zero-padding -> autocorrélation linéaire (pas circulaire)
    w = np.hanning(frame_size + 1)[:-1]
    w_ac = np.correlate(w, w, mode="full")[frame_size - 1:]
    w_ac = w_ac / w_ac[0]
    lmin, lmax = int(sr / fmax), int(sr / fmin)
    D2 = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=frame_size)
    vals, f0s = [], []
    for s in range(0, D2.shape[1], chunk):
        ac = np.fft.irfft(np.abs(D2[:, s:s + chunk]) ** 2, n=n_fft, axis=0)[:lmax + 1]
        ac = ac / (ac[0:1] + 1e-12)
        r = ac[lmin:lmax + 1] / w_ac[lmin:lmax + 1, None]
        k = np.argmax(r, axis=0)
        vals.append(r[k, np.arange(r.shape[1])])
        f0s.append(sr / (k + lmin))
    return np.concatenate(vals).astype(np.float32), np.concatenate(f0s).astype(np.float32)


def compute_features(y, sr=sr, frame_size=frame_size, hop_size=hop_size, n_mels=40):
    """Dictionnaire de caractéristiques par trame (n trames, centres à i*hop/sr comme la référence)."""
    D = librosa.stft(y, n_fft=frame_size, hop_length=hop_size, win_length=frame_size)
    S = np.abs(D)
    P = S ** 2
    F = {}
    F["rms"] = librosa.feature.rms(S=S, frame_length=frame_size)[0].astype(np.float32)
    mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=n_mels)
    F["logmel"] = np.log(mel + 1e-10).astype(np.float32)                      # (n_mels, n)
    F["flatness"] = librosa.feature.spectral_flatness(S=S)[0].astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    band = (freqs >= 250) & (freqs <= 4000)
    Pb = P[band] + 1e-12
    p = Pb / Pb.sum(0, keepdims=True)
    F["entropy"] = (-(p * np.log(p)).sum(0) / np.log(band.sum())).astype(np.float32)
    F["centroid"] = librosa.feature.spectral_centroid(S=S, sr=sr)[0].astype(np.float32)
    F["zcr"] = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_size,
                                                  center=True)[0].astype(np.float32)
    F["periodicity"], F["f0"] = periodicity(y, sr=sr, frame_size=frame_size, hop_size=hop_size)
    return F, D


def _cache_one(args):
    split, uri = args
    out = CACHE_DIR / split / f"{uri}.npz"
    if out.exists():
        return uri
    audio_dir, _ = SPLITS[split]
    wav = audio_dir / f"{uri}.wav"
    y = load_audio(wav)
    F, D = compute_features(y)
    del y
    energy_vad_ref = import_energy_vad_ref()
    ref_tl = energy_vad_ref(D, sr=sr, frame_size=frame_size, hop_size=hop_size, uri=uri, **REF_PARAMS)
    F["ref_vad"] = np.array([[s.start, s.end] for s in ref_tl], dtype=np.float64).reshape(-1, 2)
    F["duration"] = np.array(sf.info(wav).duration)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **F)
    return uri


def build_cache(split, rttm, workers=3):
    uris = sorted(load_rttm(rttm).keys())
    with Pool(workers) as pool:
        for i, uri in enumerate(pool.imap_unordered(_cache_one, [(split, u) for u in uris]), 1):
            print(f"[{i}/{len(uris)}] {uri}", flush=True)


_MEM = {}


def load_cache(split, uri):
    key = (split, uri)
    if key not in _MEM:
        with np.load(CACHE_DIR / split / f"{uri}.npz") as z:
            _MEM[key] = {k: z[k] for k in z.files}
    return _MEM[key]


_REFS = {}


def load_ref(split, uri):
    key = (split, uri)
    if key not in _REFS:
        _REFS[key] = load_rttm(SPLITS[split][1] / f"{uri}.rttm")[uri]
    return _REFS[key]


def frame_labels(split, uri, n):
    """Vérité terrain par trame (1 si le centre de la trame est dans la parole de référence)."""
    t = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_size)
    lab = np.zeros(n, dtype=np.int8)
    for seg in load_ref(split, uri).get_timeline().support():
        lab[(t >= seg.start) & (t < seg.end)] = 1
    return lab


# ---------------- post-traitement commun ----------------
def frames_to_timeline(is_speech, min_silence=0.5, min_speech=0.25, uri=None, hop_size=hop_size):
    """Même post-traitement que energy_vad : trames -> segments, comble les trous < min_silence,
    puis supprime la parole < min_speech."""
    d = np.diff(np.asarray(is_speech, dtype=int), prepend=0, append=0)
    t_s = librosa.frames_to_time(np.flatnonzero(d == 1), sr=sr, hop_length=hop_size)
    t_e = librosa.frames_to_time(np.flatnonzero(d == -1), sr=sr, hop_length=hop_size)
    tl = Timeline([Segment(s, e) for s, e in zip(t_s, t_e)], uri=uri)
    tl = tl.support(collar=min_silence)
    return Timeline([seg for seg in tl if seg.duration >= min_speech], uri=uri)


def adjust_bounds(tl, d_start=0.0, d_end=0.0, uri=None):
    """Déplace les frontières : début + d_start, fin - d_end (valeurs > 0 = rétrécir le segment)."""
    out = [Segment(s.start + d_start, s.end - d_end) for s in tl if s.duration > d_start + d_end]
    return Timeline(out, uri=uri)


# ---------------- référence (sortie d'energy_vad mise en cache) et ré-implémentation de contrôle ----------------
def vad_reference(F, uri=None):
    return Timeline([Segment(s, e) for s, e in F["ref_vad"]], uri=uri)


def vad_energy(F, uri=None, seuil_db=-30.0, delta_db=3.18, smooth_frames=3, min_silence=0.5, min_speech=0.25,
               level_ref="max", pct=50.0):
    """Ré-implémentation paramétrable d'energy_vad sur le RMS mis en cache (pour les balayages).
    level_ref="max" : dB relatifs au max du fichier (= energy_vad) ;
    level_ref="pct" : dB relatifs au percentile `pct` de l'énergie du fichier (normalisation robuste)."""
    x = librosa.amplitude_to_db(F["rms"], ref=np.max)
    if level_ref == "pct":
        x = x - np.percentile(x, pct)
    if smooth_frames > 1:
        x = uniform_filter1d(x, size=smooth_frames, mode="nearest")
    s = hysteresis(x, seuil_haut=seuil_db + delta_db / 2, seuil_bas=seuil_db - delta_db / 2)
    return frames_to_timeline(s, min_silence, min_speech, uri)


# ---------------- 1er étage : classifieur trame (HGB) appris sur nos données ----------------
def dev_train_uris():
    """Fichiers dev hors du sample de réglage : seuls fichiers utilisés pour l'apprentissage."""
    tune = set(load_rttm(TUNE_RTTM).keys())
    return sorted(u for u in load_rttm(ALL_DEV_RTTM).keys() if u not in tune)


def train_items(train="devtrain"):
    """Jeu d'entraînement : "devtrain" = 126 fichiers dev hors sample90 (évaluable sur sample90),
    "alldev" = les 216 fichiers dev (production)."""
    if train == "devtrain":
        return [("dev", u) for u in dev_train_uris()]
    if train == "alldev":
        return [("dev", u) for u in sorted(load_rttm(ALL_DEV_RTTM).keys())]
    raise ValueError(train)


def _mstd(x, w):
    """Moyenne et écart-type glissants (centrés, w trames) le long du dernier axe."""
    m = uniform_filter1d(x, w, axis=-1, mode="nearest")
    m2 = uniform_filter1d(x * x, w, axis=-1, mode="nearest")
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def build_X(F, feats="base"):
    """Matrice (n trames, d) de caractéristiques normalisées par fichier.
    feats="base" : log-mel normalisé + niveaux relatifs + descripteurs spectraux + contexte (11 et 31 trames).
    feats="base+local" : + creux d'énergie locaux (pauses courtes). C'est la version retenue."""
    lm = F["logmel"].astype(np.float64)
    n = lm.shape[1]
    lm_n = lm - np.median(lm, axis=1, keepdims=True)              # normalisation par bande et par fichier
    e = np.log(F["rms"].astype(np.float64) + 1e-10)
    e_hi = e - np.percentile(e, 95)                                 # niveau relatif au "fort" du fichier
    e_lo = e - np.percentile(e, 5)                                  # niveau relatif au plancher de bruit
    desc = np.vstack([np.log(F["flatness"] + 1e-10), F["entropy"], F["centroid"] / 1000.0,
                      F["zcr"] * 10, F["periodicity"]]).astype(np.float64)
    rows = [lm_n, e_hi[None], e_lo[None], desc]
    for w in (11, 31):                                             # ~0,36 s et ~1 s de contexte
        m, s = _mstd(np.vstack([e_hi, desc[4], desc[1], desc[0]]), w)
        rows += [m, s]
        _, s_lm = _mstd(lm_n, w)                                   # variabilité spectrale long terme
        rows.append(s_lm.mean(0, keepdims=True))
    m11, _ = _mstd(lm_n, 11)
    rows.append(m11)
    opts = feats.split("+")[1:]
    if "local" in opts:
        # creux d'énergie LOCAUX (pauses courtes) : niveau relatif au voisinage proche, pentes
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        mel_f = librosa.mel_frequencies(n_mels=lm.shape[0] + 2, fmax=sr / 2)[1:-1]
        sb = (mel_f >= 300) & (mel_f <= 3400)
        e_sb = np.log(np.exp(lm[sb]).sum(0) + 1e-10)                # énergie bande parole 300-3400 Hz
        for x in (e, e_sb):
            rows += [(x - maximum_filter1d(x, 31, mode="nearest"))[None],
                     (x - maximum_filter1d(x, 9, mode="nearest"))[None],
                     (x - uniform_filter1d(x, 11, mode="nearest"))[None],
                     (x - uniform_filter1d(x, 31, mode="nearest"))[None],
                     (minimum_filter1d(x, 5, mode="nearest") - uniform_filter1d(x, 31, mode="nearest"))[None],
                     np.gradient(x)[None]]
        rows.append((desc[4] - uniform_filter1d(desc[4], 31, mode="nearest"))[None])
    return np.vstack(rows).T.astype(np.float32)


def training_set(items, feats="base+local", step=4):
    Xs, ys = [], []
    for sp, u in items:
        F = load_cache(sp, u)
        X = build_X(F, feats)
        y = frame_labels(sp, u, X.shape[0])
        Xs.append(X[::step])
        ys.append(y[::step])
    return np.vstack(Xs), np.concatenate(ys)


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)


def _tag(train):
    return "" if train == "devtrain" else f"_{train}"


_MODELS = {}


def get_model(feats="base+local", step=4, train="devtrain"):
    """1er étage : HGB parole/non-parole, entraîné (ou relu depuis results/vad_agent/models/) sur `train`."""
    import pickle
    key = ("stage1", feats, step, train)
    if key in _MODELS:
        return _MODELS[key]
    path = AGENT_DIR / "models" / (f"hgb_{feats}_s{step}" + ("" if train == "devtrain" else f"_train{train}") + ".pkl")
    if path.exists():
        with open(path, "rb") as f:
            _MODELS[key] = pickle.load(f)
        return _MODELS[key]
    X, y = training_set(train_items(train), feats, step)
    model = _hgb().fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    _MODELS[key] = model
    return model


def _logodds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def speech_llr(F, feats="base+local", step=4, train="devtrain"):
    """Log-odds parole/non-parole du 1er étage par trame (mis en cache dans F)."""
    ck = f"_llr_{feats}_{step}_{train}"
    if ck not in F:
        F[ck] = _logodds(get_model(feats, step, train).predict_proba(build_X(F, feats))[:, 1])
    return F[ck]


# ---------------- 2e étage : empilement temporel sur le contexte des log-odds du 1er ----------------
LLR_OFFSETS = (-20, -12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 20)


def stack_X(F, llr, feats1):
    """Entrée du 2e étage : log-odds du 1er étage décalés (±20 trames) + min/max/moyenne glissants + X du 1er étage."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    n = len(llr)
    rows = [llr[np.clip(np.arange(n) + o, 0, n - 1)] for o in LLR_OFFSETS]
    for w in (5, 11, 31):
        rows += [uniform_filter1d(llr, w, mode="nearest"), maximum_filter1d(llr, w, mode="nearest"),
                 minimum_filter1d(llr, w, mode="nearest")]
    return np.hstack([np.vstack(rows).T.astype(np.float32), build_X(F, feats1)])


def oof_llr(feats1="base+local", folds=3, step=4, train="devtrain"):
    """Log-odds HORS-ÉCHANTILLON du 1er étage sur le jeu d'entraînement (validation croisée par fichier, 3 plis :
    pli k = fichiers d'indice i tel que i % 3 == k dans la liste triée)."""
    path = AGENT_DIR / "models" / f"oof_hgb_{feats1}_f{folds}_s{step}{_tag(train)}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    items = train_items(train)
    out = {}
    for k in range(folds):
        test_it = items[k::folds]
        train_it = [it for i, it in enumerate(items) if i % folds != k]
        X, y = training_set(train_it, feats1, step)
        m = _hgb().fit(X, y)
        del X, y
        for sp, u in test_it:
            out[u] = _logodds(m.predict_proba(build_X(load_cache(sp, u), feats1))[:, 1]).astype(np.float32)
        print(f"oof pli {k + 1}/{folds}", flush=True)
    np.savez(path, **out)
    return out


def get_stack_model(feats1="base+local", step=4, train="devtrain"):
    """2e étage : HGB sur stack_X, entraîné sur les log-odds hors-échantillon du 1er étage."""
    import pickle
    key = ("stack", feats1, step, train)
    if key in _MODELS:
        return _MODELS[key]
    path = AGENT_DIR / "models" / f"stack_hgb_{feats1}_s{step}{_tag(train)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            _MODELS[key] = pickle.load(f)
        return _MODELS[key]
    oof = oof_llr(feats1, step=step, train=train)
    Xs, ys = [], []
    for sp, u in train_items(train):
        X = stack_X(load_cache(sp, u), oof[u], feats1)
        Xs.append(X[::step])
        ys.append(frame_labels(sp, u, X.shape[0])[::step])
    model = _hgb().fit(np.vstack(Xs), np.concatenate(ys))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    _MODELS[key] = model
    return model


def vad_stack(F, uri=None, feats1="base+local", step=4, seuil=0.5, min_silence=0.5, min_speech=0.25,
              d_end=-0.1, train="devtrain"):
    """VAD RETENUE (itération 12) : HGB trame (1er étage) -> HGB sur le contexte temporel des log-odds (2e étage)
    -> seuil -> min_silence / min_speech -> fins de segments prolongées de |d_end| s.
    train="devtrain" (évaluable sur sample90) ou "alldev" (production)."""
    ck = f"_stack_{feats1}_{step}{_tag(train)}"
    if ck not in F:
        llr1 = speech_llr(F, feats1, step, train)
        F[ck] = _logodds(get_stack_model(feats1, step, train).predict_proba(stack_X(F, llr1, feats1))[:, 1])
    tl = frames_to_timeline(F[ck] > seuil, min_silence, min_speech, uri)
    if d_end:
        tl = adjust_bounds(tl, 0.0, d_end, uri)
    return tl


# configuration retenue : évaluation (dev-train) et production (216 fichiers dev)
BEST_PARAMS = dict(seuil=0.5, d_end=-0.1, min_silence=0.5, min_speech=0.25, train="devtrain")
PROD_PARAMS = dict(BEST_PARAMS, train="alldev")


def best_vad(y, uri=None, **kw):
    """MEILLEURE VAD, branchable : signal 16 kHz (mono) -> Timeline des zones de parole.
    vad_stack avec les modèles de PRODUCTION (entraînés sur les 216 fichiers dev, results/vad_agent/models/*alldev*).
    kw surcharge PROD_PARAMS (ex. train="devtrain")."""
    F, _ = compute_features(y)
    return vad_stack(F, uri=uri, **{**PROD_PARAMS, **kw})


METHODS = {"reference": vad_reference, "energy": vad_energy, "stack": vad_stack,
           "best": lambda F, uri=None, **kw: vad_stack(F, uri=uri, **{**BEST_PARAMS, **kw})}


# ---------------- évaluation ----------------
def evaluate(method, split="dev", rttm="samples/dev/sample90_seed130.rttm", params=None, verbose=False):
    """Renvoie (dict résumé, DataFrame par fichier). Résumé : FA %, MISS %, detection error %."""
    fn = METHODS[method] if isinstance(method, str) else method
    params = params or {}
    uris = sorted(load_rttm(rttm).keys())
    metric = DetectionErrorRate(collar=0.0, skip_overlap=False)
    rows = []
    for uri in uris:
        F = load_cache(split, uri)
        ref = load_ref(split, uri)
        uem = Timeline([Segment(0, float(F["duration"]))])
        hyp = fn(F, uri=uri, **params)
        d = metric(ref, hyp.to_annotation(), uem=uem, detailed=True)
        rows.append(dict(uri=uri, FA=d["false alarm"], MISS=d["miss"], total=d["total"],
                         duration=float(F["duration"])))
    df = pd.DataFrame(rows)
    acc = metric.accumulated_
    res = {"FA_%": 100 * acc["false alarm"] / acc["total"], "MISS_%": 100 * acc["miss"] / acc["total"],
           "detection_error_%": 100 * abs(metric), "FA_s": acc["false alarm"], "MISS_s": acc["miss"],
           "speech_s": acc["total"]}
    return res, df


def fmt(res):
    return (f"FA {res['FA_%']:.3f} % | Miss {res['MISS_%']:.3f} % | critère {res['detection_error_%']:.3f} % "
            f"(FA {res['FA_s']:.1f} s, Miss {res['MISS_s']:.1f} s, parole {res['speech_s']:.1f} s)")


def _parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(v, v)


def cli():
    ap = argparse.ArgumentParser(description="VAD agent")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("cache")
    c.add_argument("--split", default="dev")
    c.add_argument("--rttm", default="samples/dev/sample90_seed130.rttm")
    c.add_argument("--workers", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("--method", default="reference", choices=sorted(METHODS))
    e.add_argument("--split", default="dev")
    e.add_argument("--rttm", default="samples/dev/sample90_seed130.rttm")
    e.add_argument("--set", nargs="*", default=[], help="param=valeur fixe")
    e.add_argument("--sweep", default=None, help="param=v1,v2,... (balayage d'UN paramètre)")
    e.add_argument("--per-file", default=None, help="csv par fichier (dans results/vad_agent/)")
    a = ap.parse_args()
    if a.cmd is None:
        main()                                    # comportement historique : balayage seuil/delta
    elif a.cmd == "cache":
        build_cache(a.split, a.rttm, a.workers)
    elif a.cmd == "eval":
        fixed = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in a.set)}
        grid = [{}]
        if a.sweep:
            k, vs = a.sweep.split("=", 1)
            grid = [{k: _parse_value(v)} for v in vs.split(",")]
        for g in grid:
            p = {**fixed, **g}
            res, df = evaluate(a.method, a.split, a.rttm, p)
            print(f"{a.method} {p} : {fmt(res)}", flush=True)
            if a.per_file:
                df.to_csv(AGENT_DIR / a.per_file, index=False, float_format="%.3f")


if __name__ == "__main__":
    cli()
