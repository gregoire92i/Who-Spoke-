import numpy as np
import librosa
from pyannote.core import Segment, Timeline, Annotation
from embeddings import timeline_to_mask
#parameters by default
from audio_features import sr, frame_size, hop_size
#print(sr, frame_size, hop_size)   # 16000 1024 160



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
                          sr=sr, hop_size=hop_size, return_info=False, ref=None, mapping=None,
                          fill_unlabeled=False):
    """Post-traitement option B : labels de fenêtres -> Annotation pyannote (sans lissage).
 
    labels : (n_fenetres,) sortie de spectral_clustering
    X : (n_fenetres, dim) sortie de compute_embeddings (sert au départage)
    win_segs : liste de Segment, sortie de compute_embeddings
    vad : Timeline de energy_vad ; n_frames : mfcc.shape[1]
    ref, mapping : ORACLE vote (diagnostic), None par défaut = comportement normal.
      ref : Annotation de référence ; mapping : dict {label hyp "spkXX": label ref},
      sortie de DiarizationErrorRate().optimal_mapping(ref, hyp) calculée sur la 1re passe.
    fill_unlabeled : False par défaut = comportement d'origine (trames de parole sans fenêtre laissées sans label).
      True : étape 4 ci-dessous (piste 1 du subagent confusion-improver, results/confusion_improver/journal/tour_01.md).
 
    Pour chaque trame de 10 ms :
      1. vote : chaque fenêtre qui couvre la trame vote pour son label ;
      2. départage : si plusieurs labels ont le même nombre de votes, on garde celui
         de la fenêtre la plus proche du centroïde de son cluster (window_scores) ;
      2bis. ORACLE vote (si ref et mapping) : si le label choisi ne correspond pas au vrai locuteur
         mais qu'une fenêtre qui couvre la trame porte le bon label, on prend ce label
         (meilleur score si plusieurs) ; sinon la décision normale est gardée ;
      3. masquage : une trame hors VAD ne reçoit aucun label ;
         une trame de parole couverte par aucune fenêtre non plus (elle comptera en missed) ;
      4. remplissage (si fill_unlabeled) : chaque trame de parole sans label reçoit le label de la trame
         labellisée la plus proche du même segment de VAD (égalité -> la plus à gauche) ; si le segment
         n'a aucune trame labellisée (segment trop court pour une fenêtre), la plus proche de tout le fichier.
         Fait après le vote oracle et le masquage : les trames remplies copient le label final (éventuellement
         corrigé par l'oracle) de leur voisine ; le vote oracle ne s'applique pas à elles (aucune fenêtre ne
         les couvre) et elles ne comptent pas dans n_oracle_changed. Aucune trame hors VAD n'est remplie.
    Les trames consécutives de même label forment un segment.
 
    Renvoie une Annotation (labels "spk00", "spk01"...) et, si return_info=True, un dict :
      frame_labels (n_frames,) : label par trame, -1 = aucun
      n_speech : trames de parole (VAD)
      n_uncovered : trames de parole sans label à la fin (couvertes par aucune fenêtre ; 0 avec fill_unlabeled,
        sauf fichier sans aucune trame labellisée)
      n_filled : trames de parole remplies par l'étape 4 (0 si fill_unlabeled=False)
      n_tie : trames de parole où le départage par score a été utilisé
      n_oracle_changed : trames de parole dont le label a été changé par le vote oracle (0 sinon)"""
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
 
    # ORACLE vote : corrige le choix du label là où une fenêtre qui couvre la trame porte le bon locuteur
    oracle_changed = np.zeros(n_frames, dtype=bool)
    if ref is not None and mapping is not None and k > 0:
        # ref_mask[t, l] = True si le locuteur de la référence associé au cluster l parle à la trame t
        # (clusters absents du mapping = clusters en trop : colonne False, jamais "bons")
        ref_mask = np.zeros((n_frames, k), dtype=bool)
        for hyp_label, ref_label in mapping.items():
            l = int(str(hyp_label)[3:])                 # "spk03" -> 3 (format écrit plus bas)
            if l < k:
                ref_mask[:, l] = timeline_to_mask(ref.label_timeline(ref_label), n_frames,
                                                  sr=sr, hop_size=hop_size)
        # ok[t, l] : label l porté par au moins une fenêtre qui couvre t ET associé au vrai locuteur de t
        ok = (votes > 0) & ref_mask
        # label actuel déjà bon ? (en chevauchement, deux labels peuvent être bons : on ne touche pas)
        current_ok = (frame_labels >= 0) & ok[np.arange(n_frames), np.clip(frame_labels, 0, None)]
        fix = ok.any(axis=1) & ~current_ok              # trames corrigeables par le vote oracle
        frame_labels[fix] = np.argmax(np.where(ok, best, -np.inf), axis=1)[fix]
        oracle_changed = fix
 
    # masquage par la VAD (en mode concat, une fenêtre peut enjamber un silence)
    mask = timeline_to_mask(vad, n_frames, sr=sr, hop_size=hop_size)
    frame_labels[~mask] = -1
    if k > 0:
        n_tie = int((tie & mask).sum())
    n_oracle_changed = int((oracle_changed & mask).sum())   # parole uniquement, comme n_tie

    # remplissage : trame de parole sans label -> label de la trame labellisée la plus proche
    n_filled = 0
    labeled_all = np.flatnonzero(frame_labels >= 0)
    if fill_unlabeled and len(labeled_all) > 0:
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_size)  # comme timeline_to_mask
        filled = frame_labels.copy()
        for seg in vad:
            idx = np.flatnonzero((times >= seg.start) & (times < seg.end))   # trames du segment de VAD
            unl = idx[frame_labels[idx] < 0]
            if len(unl) == 0:
                continue
            lab = idx[frame_labels[idx] >= 0]
            src = lab if len(lab) else labeled_all          # priorité au même segment de VAD
            pos = np.searchsorted(src, unl)
            left = src[np.clip(pos - 1, 0, len(src) - 1)]
            right = src[np.clip(pos, 0, len(src) - 1)]
            nearest = np.where(np.abs(unl - left) <= np.abs(right - unl), left, right)
            filled[unl] = frame_labels[nearest]
        n_filled = int(((filled >= 0) & (frame_labels < 0)).sum())
        frame_labels = filled
 
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
                "n_tie": n_tie,
                "n_oracle_changed": n_oracle_changed,
                "n_filled": n_filled}
        return annotation, info
    return annotation