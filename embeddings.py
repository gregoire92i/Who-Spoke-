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
#embeddings parameters
windows_size=1.0

def timeline_to_mask(tl, n_frames, sr=sr, hop_size=hop_size):
    """prends en entrée la timeline tl sortie de vad, masque booléen (n_frames,), renvoie True si la trame est dans un segment de parole."""
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_size)#gives the time for each frame
    mask = np.zeros(n_frames, dtype=bool)#ttes les tram à false au depart
    for seg in tl:
        mask |= (times >= seg.start) & (times < seg.end)# |= -> ou cumulatif : remplace les valeurs false par les valeurs trues calculés par cette ligne
    return mask

def sliding_windows(n_frames, win_frames, hop_frames, min_frames):
    """Bornes (début, fin) des fenêtres sur une suite de n_frames trames.
    Si la suite est plus courte que la fenêtre mais d'au moins min_frames trames,
    on fait une seule fenêtre plus courte. En dessous de min_frames, aucune fenêtre."""
    if n_frames < min_frames:#segment trop court : pas assez de trames pour des statistiques fiables
        return []
    if n_frames < win_frames:#segment entre min_frames et win_frames : une seule fenêtre qui couvre tout le segment
        return [(0, n_frames)]
    #cas normal : débuts espacés de hop_frames, chaque fenêtre fait win_frames trames
    #le +1 permet d'inclure la dernière fenêtre qui finit pile à n_frames
    #les trames de la fin qui ne remplissent pas une fenêtre complète sont ignorées (au plus hop_frames-1 trames)
    return [(s, s + win_frames) for s in range(0, n_frames - win_frames + 1, hop_frames)]

def window_stats(frames, stats="mean_std"):
    """frames : (n_mfcc, T) -> embedding (n_mfcc,) si 'mean', (2*n_mfcc,) si 'mean_std'."""
    mu = frames.mean(axis=1)#axis=1 : moyenne sur le temps (les T trames), une valeur par coefficient -> timbre moyen
    if stats == "mean":
        return mu
    #std sur le temps : variabilité de chaque coefficient (intonation, débit)
    #concatenate colle les deux vecteurs bout à bout : [mu_0..mu_12, sigma_0..sigma_12]
    return np.concatenate([mu, frames.std(axis=1)])

def compute_embeddings(mfcc, vad, win_s=windows_size, hop_s=0.5, min_s=0.5, stats="mean_std",
                       drop_c0=False, mode="segment", sr=sr, hop_size=hop_size):
    """Un embedding par fenêtre glissante, sur les trames de parole uniquement (D-55).
    mfcc : (n_mfcc, n_frames) de compute_mfcc ; vad : Timeline de energy_vad.
    mode 'segment' : on fait glisser la fenêtre dans chaque segment de parole.
    mode 'concat'  : on retire le silence, on concatène, puis on fait glisser la fenêtre.
    Renvoie X (n_fenetres, dim) et la liste des Segment (en secondes) de chaque fenêtre."""
    #garde-fous : une faute de frappe dans mode ou stats doit planter, pas passer silencieusement
    if mode not in ("segment", "concat"):
        raise ValueError(f"mode doit être 'segment' ou 'concat', reçu {mode}")
    if stats not in ("mean", "mean_std"):
        raise ValueError(f"stats doit être 'mean' ou 'mean_std', reçu {stats}")

    #conversion secondes -> nombre de trames : 1 trame MFCC = hop_size/sr = 10 ms
    #round puis int : évite qu'une erreur d'arrondi (99.9999) donne 99 au lieu de 100
    win = int(round(win_s * sr / hop_size))      # 1,0 s -> 100 trames
    hop = int(round(hop_s * sr / hop_size))      # 0,5 s -> 50 trames
    min_f = int(round(min_s * sr / hop_size))    # 0,5 s -> 50 trames

    if drop_c0:
        mfcc = mfcc[1:]#retire la ligne 0 (c0), qui suit surtout le volume et dépend de la distance au micro
    n_frames = mfcc.shape[1]
    #temps (en s) du début de chaque trame, même convention que energy_vad et timeline_to_mask
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_size)
    frame_dur = hop_size / sr#durée d'une trame (0,01 s), pour calculer la fin de la dernière trame d'une fenêtre

    #groupes = listes d'indices de trames sur lesquelles la fenêtre va glisser
    if mode == "segment":
        #un groupe par segment de parole : la fenêtre ne traverse jamais un silence
        groups = [np.flatnonzero((times >= seg.start) & (times < seg.end)) for seg in vad]
    else:
        #un seul groupe : toutes les trames de parole bout à bout (le silence est retiré)
        #flatnonzero transforme le masque booléen en liste d'indices des trames True
        groups = [np.flatnonzero(timeline_to_mask(vad, n_frames, sr=sr, hop_size=hop_size))]

    X, win_segs = [], []#X : un embedding par fenêtre ; win_segs : la position de chaque fenêtre en secondes
    for idx in groups:
        #a, b sont des positions dans idx (pas des numéros de trame)
        for a, b in sliding_windows(len(idx), win, hop, min_f):
            frames = idx[a:b]#vrais numéros de trame de cette fenêtre (pas forcément consécutifs en mode concat)
            X.append(window_stats(mfcc[:, frames], stats))#mfcc[:, frames] : tous les coefficients, seulement ces trames -> (n_mfcc, T)
            #début = temps de la 1re trame, fin = temps de la dernière trame + sa durée
            #en mode concat, ce Segment peut couvrir un silence entre deux segments de parole
            win_segs.append(Segment(times[frames[0]], times[frames[-1]] + frame_dur))

    dim = mfcc.shape[0] * (1 if stats == "mean" else 2)#taille d'un embedding : n_mfcc ou 2*n_mfcc
    #np.array empile la liste en matrice (n_fenetres, dim)
    #si aucune fenêtre (fichier sans parole, cas pqmho), on renvoie quand même une matrice vide de la bonne largeur
    X = np.array(X) if X else np.empty((0, dim))
    return X, win_segs