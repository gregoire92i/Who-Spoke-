import librosa, librosa.display, soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pyannote.database.util import load_rttm
from rttm_files import load_file_annotations
from pyannote.core import Segment, Timeline, Annotation

#default parameters (cuz unless modoified in main)
sr=16000#frequence d'echantillonage
frame_size=64#en ms
hop_size=10#en ms
#conversion en nombre/s
frame_size=int(frame_size*(sr/1000))
hop_size=int(hop_size*(sr/1000))

#charger les audios
def load_audio (path,sr):
    y,sr=librosa.load(path,sr=sr,mono=True)
    return y

#calcul fft
def FFT(signal_audio,frame_size=frame_size,hop_size=hop_size):
    """#fait la tranformée de fourier rapide du signal, paramètres en ms"""
    signal_audio=signal_audio#car stft prend en parametres le nombre d'echantillons par trame
    hann_size=frame_size# on veut que recouvrir exactement la frame
    FFT_output=librosa.stft(signal_audio,n_fft=frame_size,hop_length=hop_size,win_length=hann_size)
    return FFT_output

#calcul RMS : energie de chaque trame pour le seuil de parole VAD
def RMS_on_FFT(FFT_output,frame_size=frame_size):
    rms=librosa.feature.rms(S=np.abs(FFT_output),frame_length=frame_size)
    return rms


#axe d'amelioration -> smoothing sur k trames
def energy_vad(D, sr=sr, frame_size=frame_size, hop_size=hop_size,
               seuil_db=-29.0, min_silence=0.5, min_speech=0.3, uri=None):
    """VAD par énergie avec smoothing par durées minimales (D-53, D-54).
    D : sortie de FFT() (STFT complexe). Renvoie une Timeline des zones de parole."""
    # 1. énergie par trame, en dB relatifs au max du fichier
    rms = RMS_on_FFT(D, frame_size=frame_size)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)[0]   # (1, n) -> (n,)

    # 2 décision trame par trame (0/1)
    is_speech = (rms_db > seuil_db).astype(int)#rms db>seuil_db produit un tableau de booleen, astype transforme en 1 et 0. -> Booleen pas deja 1 et 0 ?

    # 3 changements d'état : +1 = début de parole, -1 = fin
    #    prepend/append=0 gèrent un fichier qui commence ou finit en parole
    d = np.diff(is_speech, prepend=0, append=0)#np diff -> la différence entre éléments consécutifs, prepend et append un padding de 0 pour calculer le 1er element du tableau
    starts = np.flatnonzero(d == 1)#flatnonzero renvoie SEULEMENT les indices où la valeur est vraie (1), starts un tableau contenant uniquement ces indices -> creer des segments ensuite
    ends = np.flatnonzero(d == -1)# mme chose mais (-1)

    # 4 indices de trame -> secondes (sr explicite : car sr de base librosa = 22050 -> le temps en s serait faux)
    t_starts = librosa.frames_to_time(starts, sr=sr, hop_length=hop_size)#tstarts tableau de sdebuts de parole en s
    t_ends = librosa.frames_to_time(ends, sr=sr, hop_length=hop_size)
    tl = Timeline([Segment(s, e) for s, e in zip(t_starts, t_ends)], uri=uri)# zip associe 1er debut de t starts avec 1er debut de t end, marche juste avec les indices egalement

    # 5. smoothing : d'abord combler les trous courts, PUIS supprimer la parole courte
    tl = tl.support(collar=min_silence)
    tl = Timeline([seg for seg in tl if seg.duration >= min_speech], uri=uri)
    return tl

def compute_mfcc(FFT_output, sr=sr, n_mfcc=13, n_mels=40):#appeleée dans compute embeddings, -> pr pas calculer d'embedding sur des trames or parole
    """Matrice MFCC (n_mfcc, n_frames) calculée sur la STFT déjà faite par FFT()."""
    if not 13 <= n_mfcc <= 20:
        raise ValueError(f"n_mfcc doit être entre 13 et 20, reçu {n_mfcc}")
    power = np.abs(FFT_output)**2                     # spectre de puissance
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)                 # log avant la DCT
    return librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)

def save_mfcc_png(mfcc, vad, output_path, sr=sr, hop_size=hop_size, ref=None, title="MFCC", show_c0=False):
    """Sauvegarde la matrice MFCC (n_mfcc, n_frames) et la VAD (Timeline) dans un PNG.
    ref (optionnel) : Annotation de référence, affichée sous la VAD."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    first = 0 if show_c0 else 1          # c0 = énergie globale, écrase l'échelle de couleurs
    img = librosa.display.specshow(mfcc[first:], sr=sr, hop_length=hop_size, x_axis="time", ax=axes[0])
    fig.colorbar(img, ax=axes)            # ax=axes : les deux graphes restent alignés
    axes[0].set(title=title, ylabel=f"coefficients {first} à {mfcc.shape[0]-1}", xlabel="")
    for seg in vad:
        axes[1].axvspan(seg.start, seg.end, ymin=0.5, ymax=1, color="red", alpha=0.4)
    if ref is not None:
        for seg in ref.get_timeline().support():
            axes[1].axvspan(seg.start, seg.end, ymin=0, ymax=0.5, color="green", alpha=0.4)
    axes[1].set(title="VAD (rouge, haut)" + (" / référence (vert, bas)" if ref is not None else ""),
                yticks=[], xlabel="temps (s)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)