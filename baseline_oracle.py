"""Diagnostics oracle de la baseline classique de diarisation.
Même chaîne que baseline.py (VAD énergie -> MFCC -> embeddings par fenêtre glissante -> clustering spectral
-> fenêtres vers trames (option B) -> évaluation (D-30 à D-36)), mais chaque composant peut être remplacé
par la vérité terrain (RTTM) pour mesurer ce qu'il coûte au DER :
  oracle_vad  : VAD = parole de la référence
  oracle_k    : nombre de locuteurs lu dans le RTTM (au lieu de l'eigengap)
  oracle_vote : dans les trames couvertes par des windows de labels différents, on choisit le label
                qui correspond au vrai locuteur (correspondance clusters -> locuteurs par optimal_mapping)
Protocole : activer UN SEUL oracle à la fois, mêmes fichiers et mêmes métriques que la baseline.
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
from embeddings import compute_embeddings
from clustering import spectral_clustering
from post_treatment import windows_to_annotation
from metrics_and_sanity_checks import sanity_check1_ref, sanity_check2_1speaker


#### PARAMÈTRES ####
#tous les paramètres sont ici et passés EXPLICITEMENT aux fonctions :
#les valeurs par défaut de experimenations.py ne sont jamais utilisées en silence
sr = 16000#fréquence d'échantillonnage (D-50)
frame_ms, hop_ms = 65, 32.5 #D-?
frame_size = int(frame_ms * sr / 1000)#64 ms -> 1024 échantillons
hop_size = int(hop_ms * sr / 1000)#10 ms -> 160 échantillons

seuil_db = -30.0#VAD, grid search du 05/11
min_silence, min_speech = 0.5, 0.3#smoothing de la VAD (s)
n_mfcc = 13
#embedding
win_s, hop_s, min_s = 2.0, 1.0, 0.5#fenêtres des embeddings (s)
stats, drop_c0, mode = "mean_std", False, "segment" #->à verifier
collar = 0.0#D-30 : même collar pour les sanity checks et la baseline

#ORACLE : composants remplacés par la vérité terrain (un seul à True à la fois)
oracle_vad = False#True : VAD = parole de la référence (chevauchements fusionnés)
oracle_k = False#True : nombre de locuteurs lu dans le RTTM ; False : eigengap (remplace `oracle` de baseline.py)
oracle_vote = True#True : vote oracle dans windows_to_annotation (nécessite ref et mapping)

#test
# sample_rttm = "samples/test/sample90_seed130.rttm"
# audio_dir = Path("data/voxconverse/test/voxconverse_test_wav")
# out_dir = Path("results")

#dev -> adjust parameters
sample_rttm = "samples/dev/sample90_seed130.rttm"
audio_dir = Path("data/voxconverse/dev/audio")
out_dir = Path("results")
#ORACLE : le nom du run dit quels oracles sont actifs -> on n'écrase jamais les résultats de la baseline
run_name = f"baseline_oracle_vad{int(oracle_vad)}_k{int(oracle_k)}_vote{int(oracle_vote)}_sample90_seed130"


def diarize_file(path, uri, ref, uem, *, frame_size, hop_size):# * force la defintion des parametres à gauche de * dans l'appek de la fonction
    """Toute la chaîne sur UN fichier. Les tableaux lourds (audio, STFT, MFCC, matrice d'affinité)
    sont des variables locales : ils sont libérés dès que la fonction se termine pour eviter de staurer la ram
    ORACLE : ref (Annotation de référence) et uem servent aux trois oracles.
    Renvoie hyp, vad, stats_file et hyp_sans_vote (hypothèse avant vote oracle, None si oracle_vote=False)."""
    n_ref_speakers = len(ref.labels())
    #1 échantillonnage
    audio = load_audio(str(path), sr)
    #2 trames, fenêtre de Hann, STFT
    D = FFT(audio, frame_size=frame_size, hop_size=hop_size)
    del audio#plus utilisé, on le libère tout de suite pour la ram bb
    #3 VAD (ORACLE : parole de la référence, chevauchements fusionnés par support, coupée à l'UEM)
    if oracle_vad:
        vad = ref.get_timeline().support().crop(uem)
    else:
        vad = energy_vad(D, sr=sr, frame_size=frame_size, hop_size=hop_size, seuil_db=seuil_db,
                         min_silence=min_silence, min_speech=min_speech, uri=uri)
    #4 MFCC (sur la même STFT que la VAD)
    mfcc = compute_mfcc(D, sr=sr, n_mfcc=n_mfcc)
    del D#la STFT est l'objet le plus lourd (~250 Mo pour 10 min d'audio)
    #5 embeddings par fenêtre glissante
    X, win_segs = compute_embeddings(mfcc, vad, win_s=win_s, hop_s=hop_s, min_s=min_s, stats=stats,
                                     drop_c0=drop_c0, mode=mode, sr=sr, hop_size=hop_size)
    #6 clustering spectral (ORACLE k : oracle_k au lieu de `oracle`)
    labels, info = spectral_clustering(X, oracle=oracle_k, n_speakers=n_ref_speakers, return_info=True)
    #7 fenêtres -> trames -> Annotation
    hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                      sr=sr, hop_size=hop_size, return_info=True)
    #7bis ORACLE vote : 2e passe avec la correspondance clusters -> locuteurs de la référence
    hyp_sans_vote = None
    if oracle_vote:
        #métrique SÉPARÉE : optimal_mapping ne doit pas s'accumuler dans le der de main()
        mapping = DiarizationErrorRate(collar=collar, skip_overlap=False).optimal_mapping(ref, hyp, uem=uem)
        hyp_sans_vote = hyp
        hyp, post = windows_to_annotation(labels, X, win_segs, vad, n_frames=mfcc.shape[1], uri=uri,
                                          sr=sr, hop_size=hop_size, return_info=True,
                                          ref=ref, mapping=mapping)
    stats_file = {
        "k_hyp": info["k"],
        "n_windows": len(X),
        "n_speech_frames": post["n_speech"],#trames de parole selon la VAD
        "n_uncovered": post["n_uncovered"],#trames de parole sans aucune fenêtre (comptées en missed)
        "n_tie": post["n_tie"],#trames de parole où le départage par window_scores a servi
        "n_oracle_changed": post.get("n_oracle_changed", 0),#ORACLE vote : trames dont le label a changé
    }
    return hyp, vad, stats_file, hyp_sans_vote


def main():
    #ORACLE : rappel du protocole (un seul composant remplacé à la fois)
    n_oracles = oracle_vad + oracle_k + oracle_vote
    if n_oracles != 1:
        print(f"[warning] {n_oracles} oracle(s) actif(s) : le protocole prévoit d'en activer un seul à la fois")
    print(f"=== {run_name} | oracle_vad={oracle_vad} oracle_k={oracle_k} oracle_vote={oracle_vote} ===")

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
    der_sans_vote = DiarizationErrorRate(collar=collar, skip_overlap=False)#ORACLE vote : DER de la 1re passe (garde-fou)

    rows = []
    n_files = len(sample_annotations)
    with open(out_dir / f"{run_name}.rttm", "w") as f_rttm:
        for i, uri in enumerate(sorted(sample_annotations), start=1):#sorted : ordre déterministe
            ref = sample_annotations[uri]
            path = audio_dir / f"{uri}.wav"

            duration = sf.info(str(path)).duration#D-15 : lu dans l'en-tête, sans charger l'audio
            uem = Timeline([Segment(0, duration)], uri=uri)#D-33

            t0 = time.perf_counter()
            hyp, vad, stats_file, hyp_sans_vote = diarize_file(path, uri, ref, uem,
                                                               frame_size=frame_size, hop_size=hop_size)
            proc_time = time.perf_counter() - t0#chargement compris, évaluation non comprise

            row = {
                "uri": uri,
                "duration_s": duration,
                "n_ref_speakers": len(ref.labels()),
                **stats_file,#k_hyp, n_windows, n_speech_frames, n_uncovered, n_tie, n_oracle_changed
                "DER": der(ref, hyp, uem=uem),
                "DER_forgiving": der_forgiving(ref, hyp, uem=uem),
                "JER": jer(ref, hyp, uem=uem),
                "VAD_error": det(ref, vad.to_annotation(), uem=uem),
                "RTF": proc_time / duration,#D-36
            }
            #ORACLE vote, garde-fou : le vote oracle ne peut que faire baisser le DER du fichier
            if oracle_vote:
                row["DER_sans_oracle_vote"] = der_sans_vote(ref, hyp_sans_vote, uem=uem)
                if row["DER"] > row["DER_sans_oracle_vote"] + 1e-9:
                    print(f"[warning] {uri} : DER avec vote oracle ({row['DER']:.2%}) > sans ({row['DER_sans_oracle_vote']:.2%}) -> bug probable")
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
    if oracle_vote:#ORACLE
        print(f"Parole modifiée par le vote oracle : {df['n_oracle_changed'].sum() / n_speech:.2%}")
        print(f"DER moderne sans vote oracle (1re passe) : {abs(der_sans_vote):.2%}")
    print(f"\nDétail du DER moderne :\n{der.report(display=False).tail(1).T}")

    #comparaison au plancher trivial du sanity check 2
    print(f"\nPlancher trivial (1 locuteur) : DER {der_1spk:.2%} | DER forgiving {der_1spk_forgiving:.2%}")
    print(f"Baseline vs plancher          : {(der_1spk - abs(der)) * 100:+.1f} points (moderne), "
          f"{(der_1spk_forgiving - abs(der_forgiving)) * 100:+.1f} points (forgiving) "
          f"(positif = la baseline fait mieux)")


if __name__ == "__main__":
    main()