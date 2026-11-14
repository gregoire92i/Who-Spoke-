"""Baseline classique de diarisation.
VAD énergie -> MFCC -> embeddings par fenêtre glissante -> clustering spectral
-> fenêtres vers trames (option B) -> évaluation (D-30 à D-36).
Les fichiers sont traités UN PAR UN : la RAM ne contient jamais qu'un seul audio et une seule STFT."""
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
#loading pipeline functions
from audio_features import load_audio, FFT, energy_vad, compute_mfcc
from vad import best_vad
from embeddings import compute_embeddings
from clustering import spectral_clustering, nme_spectral_clustering
from post_treatment import windows_to_annotation
from metrics_and_sanity_checks import sanity_check1_ref, sanity_check2_1speaker


#### PARAMÈTRES ####
#tous les paramètres sont ici et passés EXPLICITEMENT aux fonctions :
#les valeurs par défaut de experimenations.py ne sont jamais utilisées en silence
sr = 16000#fréquence d'échantillonnage (D-50)
frame_ms, hop_ms = 65, 32.5 #D-?
frame_size = int(frame_ms * sr / 1000)
hop_size = int(hop_ms * sr / 1000)

seuil_db = -30.0#VAD, grid search du 05/11
min_silence, min_speech = 0.5, 0.25#smoothing de la VAD (s)
n_mfcc = 13
#embedding
win_s, hop_s, min_s = 2.0, 1.0, 0.5#fenêtres des embeddings (s)
stats, drop_c0, mode = "mean_std", False, "segment"
oracle = False#True : nombre de locuteurs lu dans le RTTM ; False : eigengap
min_speakers, max_speakers = 1, 20#bornes de k pour l'eigengap (ignorées si oracle)
p_neighbors, min_neighbors = 0.02, 5#raffinement : fraction / nombre minimal de voisins gardés par window
clustering_method = "nme"#amélioration n°2 : "nme" = NME-SC (p choisi par fichier) ; "fixe" = p_neighbors fixe (ancienne baseline)
nme_max_frac, nme_n_grid = 0.25, 30#NME-SC : grille de p = n_grid entiers entre min_neighbors et max_frac * n_windows
nme_laplacian = "normalized"#NME-SC : "normalized" (seul le choix de p change) ou "unnormalized" (comme l'article)
standardize = True#centrage-réduction des embeddings sur le fichier avant le cosinus
seed = 42#graine du KMeans (reproductibilité)
collar = 0.0#D-30 : même collar pour les sanity checks et la baseline
#test
# sample_rttm = "samples/test/sample90_seed130.rttm"
# audio_dir = Path("data/voxconverse/test/voxconverse_test_wav")
# out_dir = Path("results")
# run_name = "baseline_sample90_seed130"

#dev -> adjust parameters
sample_rttm = "samples/dev/sample90_seed130.rttm"
audio_dir = Path("data/voxconverse/dev/audio")
out_dir = Path("results")
run_name = "baseline_sample90_seed130"


def diarize_file(path, uri, n_ref_speakers=None, *, frame_size, hop_size):# * force la defintion des parametres à gauche de * dans l'appel de la fonction
    """Toute la chaîne sur UN fichier. Les tableaux lourds (audio, STFT, MFCC, matrice d'affinité)
    sont des variables locales : ils sont libérés dès que la fonction se termine pour eviter de staurer la ram"""
    #1 échantillonnage
    audio = load_audio(str(path), sr)
    #2 trames, fenêtre de Hann, STFT
    D = FFT(audio, frame_size=frame_size, hop_size=hop_size)
    #3 VAD
    vad = best_vad(audio, uri=uri, train="traindev")# prends directement l'audio car a ses propres parametres de trames 
    del audio#plus utilisé, on le libère tout de suite pour la ram bb
    #4 MFCC
    mfcc = compute_mfcc(D, sr=sr, n_mfcc=n_mfcc)
    del D#la STFT est l'objet le plus lourd (~250 Mo pour 10 min d'audio)
    #5 embeddings par fenêtre glissante
    X, win_segs = compute_embeddings(mfcc, vad, win_s=win_s, hop_s=hop_s, min_s=min_s, stats=stats,
                                     drop_c0=drop_c0, mode=mode, sr=sr, hop_size=hop_size)
    #6 clustering spectral
    if clustering_method == "nme":
        labels, info = nme_spectral_clustering(X, oracle=oracle, n_speakers=n_ref_speakers,
                                               min_speakers=min_speakers, max_speakers=max_speakers,
                                               min_neighbors=min_neighbors, max_frac=nme_max_frac,
                                               n_grid=nme_n_grid, laplacian=nme_laplacian,
                                               standardize=standardize, seed=seed, return_info=True)
    elif clustering_method == "fixe":
        labels, info = spectral_clustering(X, oracle=oracle, n_speakers=n_ref_speakers,
                                           min_speakers=min_speakers, max_speakers=max_speakers,
                                           p_neighbors=p_neighbors, min_neighbors=min_neighbors,
                                           standardize=standardize, seed=seed, return_info=True)
    else:
        raise ValueError(f"clustering_method doit valoir 'nme' ou 'fixe', reçu {clustering_method}")
    #7 fenêtres -> trames -> Annotation
    hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                      sr=sr, hop_size=hop_size, return_info=True)
    stats_file = {
        "k_hyp": info["k"],
        "p_neighbors_used": info.get("p"),#NME-SC : nombre de voisins choisi pour ce fichier (None si p fixe)
        "n_windows": len(X),
        "n_speech_frames": post["n_speech"],#trames de parole selon la VAD
        "n_uncovered": post["n_uncovered"],#trames de parole sans aucune fenêtre (comptées en missed)
        "n_tie": post["n_tie"],#trames de parole où le départage par window_scores a servi
    }
    return hyp, vad, stats_file


def main():
    #0 annotations du sample, restreintes aux fichiers dont le wav existe
    #(les sanity checks appellent sf.info : un wav manquant les ferait planter)
    sample_annotations = {}
    for uri, ann in load_rttm(sample_rttm).items():
        if (audio_dir / f"{uri}.wav").exists():
            sample_annotations[uri] = ann
        else:
            print(f"[warning] {audio_dir / f'{uri}.wav'} introuvable, fichier ignoré")
    out_dir.mkdir(parents=True, exist_ok=True)

    #0bis sanity checks : on valide le protocole d'évaluation AVANT de lancer la baseline
    col = ("diarization error rate", "%")
    print("=== Sanity check 1 : référence vs elle-même (attendu : DER = 0) ===")
    sc1, sc1_forgiving = sanity_check1_ref(sample_annotations, audio_dir, check_collar=collar, display=False)
    der_sc1, der_sc1_forgiving = sc1.loc["TOTAL", col], sc1_forgiving.loc["TOTAL", col]
    assert abs(der_sc1) < 1e-6 and abs(der_sc1_forgiving) < 1e-6, \
        f"Sanity check 1 échoué : DER {der_sc1:.2f} %, DER forgiving {der_sc1_forgiving:.2f} % (attendu 0)"

    print("=== Sanity check 2 : un seul locuteur sur tout le fichier ===")
    sc2, sc2_forgiving = sanity_check2_1speaker(sample_annotations, audio_dir, check_collar=collar, display=False)
    der_1spk = sc2.loc["TOTAL", col] / 100#report() donne des %
    der_1spk_forgiving = sc2_forgiving.loc["TOTAL", col] / 100

    #métriques (D-30 à D-35) : collar 0, UEM [0, durée du wav], agrégation micro
    #chaque appel metric(ref, hyp) renvoie le score du fichier ET l'accumule ; abs(metric) = score micro
    der = DiarizationErrorRate(collar=collar, skip_overlap=False)#DER moderne, métrique principale
    der_forgiving = DiarizationErrorRate(collar=collar, skip_overlap=True)
    jer = JaccardErrorRate(collar=collar, skip_overlap=False)
    det = DetectionErrorRate(collar=collar, skip_overlap=False)#VAD seule : (FA + missed) / parole de référence

    rows = []
    n_files = len(sample_annotations)
    with open(out_dir / f"{run_name}.rttm", "w") as f_rttm:
        for i, uri in enumerate(sorted(sample_annotations), start=1):#sorted : ordre déterministe
            ref = sample_annotations[uri]
            path = audio_dir / f"{uri}.wav"

            duration = sf.info(str(path)).duration#D-15 : lu dans l'en-tête, sans charger l'audio
            uem = Timeline([Segment(0, duration)], uri=uri)#D-33

            t0 = time.perf_counter()
            hyp, vad, stats_file = diarize_file(path, uri, n_ref_speakers=len(ref.labels()),
                                       frame_size=frame_size, hop_size=hop_size)
            proc_time = time.perf_counter() - t0#chargement compris, évaluation non comprise

            row = {
                "uri": uri,
                "duration_s": duration,
                "n_ref_speakers": len(ref.labels()),
                **stats_file,#k_hyp, n_windows, n_speech_frames, n_uncovered, n_tie
                "DER": der(ref, hyp, uem=uem),
                "DER_forgiving": der_forgiving(ref, hyp, uem=uem),
                "JER": jer(ref, hyp, uem=uem),
                "VAD_error": det(ref, vad.to_annotation(), uem=uem),
                "RTF": proc_time / duration,#D-36
            }
            rows.append(row)
            hyp.write_rttm(f_rttm)#hypothèses sauvegardées : on pourra réévaluer sans tout relancer
            print(f"[{i}/{n_files}] {uri} | DER {row['DER']:.1%} | k {row['k_hyp']} (réf {row['n_ref_speakers']}) "
                  f"| RTF {row['RTF']:.3f}")

    #résultats par fichier, + ligne d'agrégat micro
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}_par_fichier.csv", index=False)
    total_time = (df["RTF"] * df["duration_s"]).sum()
    print(f"\n=== {run_name} : {len(df)} fichiers, agrégation micro ===")
    print(f"DER moderne   : {abs(der):.2%}")
    print(f"DER forgiving : {abs(der_forgiving):.2%}")
    print(f"JER           : {abs(jer):.2%}")
    print(f"VAD error     : {abs(det):.2%}")
    print(f"RTF global    : {total_time / df['duration_s'].sum():.3f}")
    #micro aussi : on somme les trames de tous les fichiers avant de diviser
    n_speech = df["n_speech_frames"].sum()
    print(f"Parole VAD sans label : {df['n_uncovered'].sum() / n_speech:.2%}")
    print(f"Parole départagée     : {df['n_tie'].sum() / n_speech:.2%}")
    print(f"\nDétail du DER moderne :\n{der.report(display=False).tail(1).T}")

    #comparaison au plancher trivial du sanity check 2
    print(f"\nPlancher trivial (1 locuteur) : DER {der_1spk:.2%} | DER forgiving {der_1spk_forgiving:.2%}")
    print(f"Baseline vs plancher          : {(der_1spk - abs(der)) * 100:+.1f} points (moderne), "
          f"{(der_1spk_forgiving - abs(der_forgiving)) * 100:+.1f} points (forgiving) "
          f"(positif = la baseline fait mieux)")


if __name__ == "__main__":
    main()