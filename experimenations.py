import librosa, librosa.display, soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pyannote.database.util import load_rttm
from rttm_files import load_file_annotations
from pyannote.core import Segment, Timeline, Annotation
from sklearn.cluster import KMeans

#parameters
sr=16000#frequence d'echantillonage
frame_size=64#en ms
hop_size=10#en ms
#conversion en nombre/s
frame_size=int(frame_size*(sr/1000))
hop_size=int(hop_size*(sr/1000))

#charger la data
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

def compute_mfcc(FFT_output, sr=sr, n_mfcc=13, n_mels=40):
    """Matrice MFCC (n_mfcc, n_frames) calculée sur la STFT déjà faite par FFT()."""
    if not 13 <= n_mfcc <= 20:
        raise ValueError(f"n_mfcc doit être entre 13 et 20, reçu {n_mfcc}")
    power = np.abs(FFT_output)**2                     # spectre de puissance
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)                 # log avant la DCT
    return librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)

def timeline_to_mask(tl, n_frames, sr=sr, hop_size=hop_size):
    """Masque booléen (n_frames,) : True si la trame est dans un segment de parole."""
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_size)
    mask = np.zeros(n_frames, dtype=bool)
    for seg in tl:
        mask |= (times >= seg.start) & (times < seg.end)
    return mask

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

def compute_embeddings(mfcc, vad, win_s=1.0, hop_s=0.5, min_s=0.5, stats="mean_std",
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


def spectral_clustering(X, oracle=False, n_speakers=None, p_neighbors=0.1, min_neighbors=5,
                        min_speakers=1, max_speakers=10, standardize=True,
                        seed=42, return_info=False):
    """Clustering spectral des embeddings, nombre de locuteurs estimé par eigengap.
 
    X : (n_fenetres, dim), sortie de compute_embeddings.
    oracle : False (défaut) -> k estimé par eigengap entre min_speakers et max_speakers.
             True -> k = n_speakers, qui doit alors être donné (lu dans le RTTM).
    p_neighbors : fraction des fenêtres gardées comme voisines de chaque fenêtre (raffinement).
    min_neighbors : nombre minimal de voisins gardés, quelle que soit la taille du fichier.
    standardize : centre-réduit chaque dimension sur le fichier avant le cosinus.
    seed : graine du k-means, pour la reproductibilité.
    Renvoie labels (n_fenetres,) entiers 0..k-1, et si return_info=True un dict
    {"k", "eigenvalues", "affinity"} pour tracer la matrice et les valeurs propres."""
 
    # --- garde-fous sur les paramètres ---
    if oracle and (n_speakers is None or n_speakers < 1):
        raise ValueError(f"mode oracle : n_speakers doit être un entier >= 1, reçu {n_speakers}")
    if not 0 < p_neighbors <= 1:
        raise ValueError(f"p_neighbors doit être dans ]0, 1], reçu {p_neighbors}")
    if not 1 <= min_speakers <= max_speakers:
        raise ValueError(f"il faut 1 <= min_speakers <= max_speakers, reçu {min_speakers}, {max_speakers}")
 
    n = X.shape[0]
    # cas dégénérés : 0 ou 1 fenêtre (fichier sans parole, cas pqmho) -> rien à regrouper
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        info = {"k": n, "eigenvalues": np.array([]), "affinity": np.ones((n, n))}
        return (labels, info) if return_info else labels
 
    # --- 1. normalisation des embeddings ---
    X = X.astype(float)
    if standardize:
        # sans ça, le cosinus est dominé par les dimensions de grande amplitude (c0, moyennes)
        # et toutes les fenêtres se ressemblent ; std nulle -> on divise par 1
        std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    # norme L2 = 1 par ligne : le produit scalaire devient la similarité cosinus
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
 
    # --- 2. matrice d'affinité : cosinus entre toutes les paires de fenêtres, dans [-1, 1] ---
    A = X @ X.T                                         # (n, n)
 
    # --- 3. raffinement : pour chaque ligne, on ne garde que les p % voisins les plus similaires ---
    # (binarisation, comme NME-SC) : 1 = « probablement le même locuteur », 0 sinon
    # min_neighbors : sur un fichier court (peu de fenêtres), p_neighbors * n donnerait 1 ou 2 voisins,
    # le graphe se casserait en dizaines de morceaux et toutes les valeurs propres vaudraient 0
    p = min(n, max(min_neighbors, int(round(p_neighbors * n))))
    # argsort décroissant par ligne, on prend les p premiers indices de chaque ligne
    top = np.argsort(-A, axis=1)[:, :p]
    A_bin = np.zeros_like(A)
    np.put_along_axis(A_bin, top, 1.0, axis=1)
    # symétrisation : le graphe doit être non orienté (i voisin de j <=> j voisin de i)
    A_bin = 0.5 * (A_bin + A_bin.T)
 
    # --- 4. Laplacien normalisé L = I - D^-1/2 A D^-1/2 ---
    deg = A_bin.sum(axis=1)                             # degré de chaque fenêtre (>0 : elle est sa propre voisine)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    L = np.eye(n) - d_inv_sqrt[:, None] * A_bin * d_inv_sqrt[None, :]
 
    # --- 5. valeurs/vecteurs propres, triés par ordre croissant (eigh : matrice symétrique) ---
    eigvals, eigvecs = np.linalg.eigh(L)
 
    # --- 6. choix de k ---
    if oracle:
        k = n_speakers
    else:
        # k groupes bien séparés -> k valeurs propres proches de 0, puis un saut
        # gaps[i] = lambda_{i+1} - lambda_i ; le plus grand saut après la k-ième valeur propre donne k
        k_max = min(max_speakers, n - 1)
        gaps = np.diff(eigvals[:k_max + 1])             # gaps[0] = saut après 1 valeur propre
        k_min = min(min_speakers, k_max)
        k = int(np.argmax(gaps[k_min - 1:k_max])) + k_min
    k = min(k, n)                                       # on ne peut pas faire plus de groupes que de fenêtres
 
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        # --- 7. chaque fenêtre est représentée par ses coordonnées sur les k premiers vecteurs propres ---
        U = eigvecs[:, :k]                              # (n, k)
        u_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.where(u_norms > 0, u_norms, 1.0)     # normalisation par ligne (Ng, Jordan, Weiss) ; ligne nulle laissée à 0
        # --- 8. k-means dans cet espace, où les groupes sont presque des points ---
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(U)
 
    if return_info:
        return labels, {"k": k, "eigenvalues": eigvals[:max_speakers + 1], "affinity": A_bin}
    return labels


##POsts traitement 

def window_scores(X, labels, standardize=True):
    """Similarité cosinus de chaque fenêtre au centroïde de son cluster.
    Même normalisation que spectral_clustering (centrage-réduction puis norme L2),
    pour que le score mesure la ressemblance dans l'espace où le clustering a travaillé.
    Renvoie scores (n_fenetres,) dans [-1, 1] : proche de 1 = fenêtre typique de son cluster."""
    X = X.astype(float)
    if standardize:
        std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
 
    scores = np.zeros(len(labels))
    for l in np.unique(labels):
        members = labels == l
        c = X[members].mean(axis=0)                     # centroïde du cluster l
        c = c / max(np.linalg.norm(c), 1e-12)           # norme 1 -> produit scalaire = cosinus
        scores[members] = X[members] @ c
    return scores
 
 
def windows_to_annotation(labels, X, win_segs, vad, n_frames, uri=None, standardize=True,
                          sr=sr, hop_size=hop_size, return_info=False):
    """Post-traitement option B : labels de fenêtres -> Annotation pyannote (sans lissage).
 
    labels : (n_fenetres,) sortie de spectral_clustering
    X : (n_fenetres, dim) sortie de compute_embeddings (sert au départage)
    win_segs : liste de Segment, sortie de compute_embeddings
    vad : Timeline de energy_vad ; n_frames : mfcc.shape[1]
 
    Pour chaque trame de 10 ms :
      1. vote : chaque fenêtre qui couvre la trame vote pour son label ;
      2. départage : si plusieurs labels ont le même nombre de votes, on garde celui
         de la fenêtre la plus proche du centroïde de son cluster (window_scores) ;
      3. masquage : une trame hors VAD ne reçoit aucun label ;
         une trame de parole couverte par aucune fenêtre non plus (elle comptera en missed).
    Les trames consécutives de même label forment un segment.
 
    Renvoie une Annotation (labels "spk00", "spk01"...) et, si return_info=True, un dict :
      frame_labels (n_frames,) : label par trame, -1 = aucun
      n_speech : trames de parole (VAD)
      n_uncovered : trames de parole couvertes par aucune fenêtre
      n_tie : trames de parole où le départage par score a été utilisé"""
    labels = np.asarray(labels)
    if len(labels) != len(win_segs) or len(labels) != X.shape[0]:
        raise ValueError(f"tailles incohérentes : {len(labels)} labels, {len(win_segs)} fenêtres, {X.shape[0]} embeddings")
 
    frame_dur = hop_size / sr
    k = int(labels.max()) + 1 if len(labels) else 0
    scores = window_scores(X, labels, standardize) if len(labels) else np.array([])
 
    # votes[t, l] : nombre de fenêtres de label l qui couvrent la trame t
    # best[t, l]  : meilleur score parmi ces fenêtres (-inf si aucune)
    votes = np.zeros((n_frames, k), dtype=int)
    best = np.full((n_frames, k), -np.inf)
    for seg, l, s in zip(win_segs, labels, scores):
        # secondes -> indices de trame ; round évite qu'une erreur d'arrondi
        # (0.29 + 0.01 = 0.30000000000000004) ajoute ou retire une trame
        a = int(round(seg.start / frame_dur))
        b = min(int(round(seg.end / frame_dur)), n_frames)
        votes[a:b, l] += 1
        best[a:b, l] = np.maximum(best[a:b, l], s)
 
    frame_labels = np.full(n_frames, -1)
    n_tie = 0
    if k > 0:
        max_votes = votes.max(axis=1)
        candidates = votes == max_votes[:, None]        # labels arrivés en tête du vote
        # parmi les labels en tête, celui de la fenêtre au meilleur score
        frame_labels = np.argmax(np.where(candidates, best, -np.inf), axis=1)
        frame_labels[max_votes == 0] = -1               # trame couverte par aucune fenêtre
        tie = (candidates.sum(axis=1) > 1) & (max_votes > 0)
 
    # masquage par la VAD (en mode concat, une fenêtre peut enjamber un silence)
    mask = timeline_to_mask(vad, n_frames, sr=sr, hop_size=hop_size)
    frame_labels[~mask] = -1
    if k > 0:
        n_tie = int((tie & mask).sum())
 
    # trames consécutives de même label -> segments
    # change[i] = True si le label change entre la trame i-1 et la trame i
    change = np.flatnonzero(np.diff(frame_labels, prepend=-2, append=-2) != 0)
    annotation = Annotation(uri=uri)
    for start, end in zip(change[:-1], change[1:]):     # run [start, end[
        l = frame_labels[start]
        if l >= 0:
            annotation[Segment(float(start * frame_dur), float(end * frame_dur))] = f"spk{l:02d}"
 
    if return_info:
        info = {"frame_labels": frame_labels,
                "n_speech": int(mask.sum()),
                "n_uncovered": int((mask & (frame_labels == -1)).sum()),
                "n_tie": n_tie}
        return annotation, info
    return annotation


# #tests MFCC
# audio1=load_audio("data/voxconverse/dev/audio/ysgbf.wav", sr)
# D = FFT(audio1)
# mfcc = compute_mfcc(D, n_mfcc=13)
# vad = energy_vad(D, seuil_db=-30.0)
# mask = timeline_to_mask(vad, mfcc.shape[1])
# mfcc_speech = mfcc[:, mask]          # (13, n_trames_parole)
# ref = load_file_annotations("data/voxconverse/annotations/dev/ysgbf.rttm")
# save_mfcc_png(mfcc, vad, "figures/mfcc_ysgbf.png", ref=ref, title="MFCC")

# #tests embedding 
# X, win_segs = compute_embeddings(mfcc, vad)
# print(X.shape, len(win_segs), win_segs[:3])


# #test clustering
# labels = spectral_clustering(X)                               # k estimé par eigengap
# labels = spectral_clustering(X, oracle=True, n_speakers=len(ref.labels()))  # mode oracle ATTENTION LES 2 ECRIVENT DANS LABELS
# labels, info = spectral_clustering(X, return_info=True)       # info["affinity"], info["eigenvalues"] pour tes figures

# #echantillonage
# audio=load_audio("data/test.wav", sr)


# print("forme :", audio.shape, "| durée :", len(audio) / sr, "s | min/max :", audio.min(), audio.max())
# print(sr)
# print(len(audio) / sr)        
# print(librosa.get_duration(y=audio, sr=sr))

# #graphique
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6),sharex=True)# nrows le nombre de ligne ncols colonne donc -> nbr de graphiques dans la figure=nrows*ncols
# librosa.display.waveshow(audio,sr=sr,ax=axes[0])
# axes[0].set(title="Forme d'onde")
# plt.savefig("figures/waveform.png")

# #trames et FFT
# D=FFT(audio)
# print(D.shape)

# #spectrogramme
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)#amplitude to db convertit en db, np.abs prend le module de D (complexe->reel), ref est valeur de ref (à l'origine ?)np.max prend la val max du spectrogramme --> important pour comparer diff fichiers entre eux qui n'ont pas dutout le meme volume

# #graphique
# spectro = librosa.display.specshow(S_db, sr=sr, hop_length=hop_size*16, x_axis="time", y_axis="hz", ax=axes[1])
# fig.colorbar(spectro, ax=axes[1], format="%+2.0f dB")
# axes[1].set(title="Spectrogramme")
# plt.tight_layout()#ajuste les marges pour eviter les chevauchements
# plt.savefig("figures/waveform.png")


# ######################
# #FICHIER2#

# #echantillonage
# audio1=load_audio("data/voxconverse/dev/audio/ysgbf.wav", sr)

# #graphique
# fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(12, 6),sharex=True)# nrows le nombre de ligne ncols colonne donc -> nbr de graphiques dans la figure=nrows*ncols
# librosa.display.waveshow(audio1,sr=sr,ax=axes1[0])
# axes1[0].set(title="Forme d'onde")

# #trames et FFT
# D1=FFT(audio1)
# print(D1.shape)
# #VAD
# rms=RMS_on_FFT(D1)

# #spectrogramme
# S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)#amplitude to db convertit en db, np.abs prend le module de D (complexe->reel), ref est valeur de ref (à l'origine ?)np.max prend la val max du spectrogramme --> important pour comparer diff fichiers entre eux qui n'ont pas dutout le meme volume

# #graphique rms
# times = librosa.times_like(rms, sr=sr, hop_length=hop_size)
# rms_db = librosa.amplitude_to_db(rms, ref=np.max)
# axes1[1].plot(times, rms_db[0])
# X=22
# axes1[1].axhline(-X, color="red", linestyle="--")# marge du seuil en dB
# axes1[1].set(title="energie")
# ref=load_file_annotations("ysgbf.rttm")
# for seg in ref.get_timeline().support():
#        axes1[1].axvspan(seg.start, seg.end, alpha=0.2, color="green")

# #graphique spectrogramme
# spectro1 = librosa.display.specshow(S_db1, sr=sr, hop_length=hop_size, x_axis="time", y_axis="hz", ax=axes1[2])
# fig1.colorbar(spectro1, ax=axes1[2], format="%+2.0f dB")
# axes1[2].set(title="Spectrogramme")
# plt.tight_layout()#ajuste les marges pour eviter les chevauchements
# plt.savefig("figures/waveform1.png")




# annot=load_rttm("data/voxconverse/annotations/dev/abjxc.rttm")["abjxc"]
# print(annot)
# print(annot.label_duration("spk00"))


