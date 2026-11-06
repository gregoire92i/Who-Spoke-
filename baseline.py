

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
 
from experimenations import (load_audio, FFT, energy_vad, compute_mfcc,
                             compute_embeddings, spectral_clustering, windows_to_annotation)
 
#### PARAMÈTRES ####
#tous les paramètres sont ici et passés EXPLICITEMENT aux fonctions :
#les valeurs par défaut de experimenations.py ne sont jamais utilisées en silence
sr = 16000#fréquence d'échantillonnage (D-50)
frame_ms, hop_ms = 64, 10#D-51
frame_size = int(frame_ms * sr / 1000)#64 ms -> 1024 échantillons
hop_size = int(hop_ms * sr / 1000)#10 ms -> 160 échantillons
 
seuil_db = -30.0#VAD, grid search du 05/11
min_silence, min_speech = 0.5, 0.3#smoothing de la VAD (s)
n_mfcc = 13
win_s, hop_s, min_s = 1.0, 0.5, 0.5#fenêtres des embeddings (s)
stats, drop_c0, mode = "mean_std", False, "segment"
oracle = False#True : nombre de locuteurs lu dans le RTTM ; False : eigengap
 
sample_rttm = "samples/sample90_seed130.rttm"
audio_dir = Path("data/voxconverse/dev/audio")
out_dir = Path("results")
run_name = "baseline_sample90_seed130"
 
 
def diarize_file(path, uri, n_ref_speakers=None):
    """Toute la chaîne sur UN fichier. Les tableaux lourds (audio, STFT, MFCC, matrice d'affinité)
    sont des variables locales : ils sont libérés dès que la fonction se termine pour eviter de staurer la ram"""
    #1 échantillonnage
    audio = load_audio(str(path), sr)
    #2 trames, fenêtre de Hann, STFT
    D = FFT(audio, frame_size=frame_size, hop_size=hop_size)
    del audio#plus utilisé, on le libère tout de suite pour la ram bb
    #3 VAD
    vad = energy_vad(D, sr=sr, frame_size=frame_size, hop_size=hop_size, seuil_db=seuil_db,
                     min_silence=min_silence, min_speech=min_speech, uri=uri)
    #4 MFCC (sur la même STFT que la VAD)
    mfcc = compute_mfcc(D, sr=sr, n_mfcc=n_mfcc)
    del D#la STFT est l'objet le plus lourd (~250 Mo pour 10 min d'audio)
    #5 embeddings par fenêtre glissante
    X, win_segs = compute_embeddings(mfcc, vad, win_s=win_s, hop_s=hop_s, min_s=min_s, stats=stats,
                                     drop_c0=drop_c0, mode=mode, sr=sr, hop_size=hop_size)
    #6 clustering spectral
    labels, info = spectral_clustering(X, oracle=oracle, n_speakers=n_ref_speakers, return_info=True)
    #7 fenêtres -> trames -> Annotation
    hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                      sr=sr, hop_size=hop_size, return_info=True)
    stats_file = {
        "k_hyp": info["k"],
        "n_windows": len(X),
        "n_speech_frames": post["n_speech"],#trames de parole selon la VAD
        "n_uncovered": post["n_uncovered"],#trames de parole sans aucune fenêtre (comptées en missed)
        "n_tie": post["n_tie"],#trames de parole où le départage par window_scores a servi
    }
    return hyp, vad, stats_file
 
 
def main():
    #0 annotations du sample : HORS de la boucle, donne la liste des fichiers à traiter
    sample_annotations = load_rttm(sample_rttm)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    #métriques (D-30 à D-35) : collar 0, UEM [0, durée du wav], agrégation micro
    #chaque appel metric(ref, hyp) renvoie le score du fichier ET l'accumule ; abs(metric) = score micro
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)#DER moderne, métrique principale
    der_forgiving = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    det = DetectionErrorRate(collar=0.0, skip_overlap=False)#VAD seule : (FA + missed) / parole de référence
 
    rows = []
    n_files = len(sample_annotations)
    with open(out_dir / f"{run_name}.rttm", "w") as f_rttm:
        for i, uri in enumerate(sorted(sample_annotations), start=1):#sorted : ordre déterministe
            ref = sample_annotations[uri]
            path = audio_dir / f"{uri}.wav"
            if not path.exists():
                print(f"[warning] {path} introuvable, fichier ignoré")
                continue
 
            duration = sf.info(str(path)).duration#D-15 : lu dans l'en-tête, sans charger l'audio
            uem = Timeline([Segment(0, duration)], uri=uri)#D-33
 
            t0 = time.perf_counter()
            hyp, vad, stats_file = diarize_file(path, uri, n_ref_speakers=len(ref.labels()))
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
 
 
if __name__ == "__main__":
    main()
# #speaking_zones=energy_vad()