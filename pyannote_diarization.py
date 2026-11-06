"""Diarisation avec la pipeline pré-entraînée pyannote/speaker-diarization-community-1.
Même protocole d'évaluation que baseline.py (D-30 à D-36), pour une comparaison directe.
Les fichiers sont traités UN PAR UN, comme dans baseline.py.

Accès au modèle (une seule fois) :
  1. accepter les conditions sur https://huggingface.co/pyannote/speaker-diarization-community-1
  2. créer un token (lecture) sur https://huggingface.co/settings/tokens
  3. dans le terminal : `hf auth login` (le token est stocké dans ~/.cache/huggingface, jamais dans le code, D-02)

Fuite de données (D-40) : segmentation-3.0 a été entraîné sur VoxConverse (split non précisé,
très probablement le dev). Un DER mesuré sur le dev est donc probablement trop optimiste."""
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import pyannote.audio
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

#### PARAMÈTRES ####
model_id = "pyannote/speaker-diarization-community-1"
oracle = False#True : nombre de locuteurs donné à la pipeline (num_speakers) ; False : estimé par la pipeline
warmup_s = 10.0#durée du passage de chauffe (s), exclu du RTF ; 0 pour désactiver

#jeu d'évaluation : décommenter UN des deux blocs
#A : sample 90 du dev, comparaison directe avec results/baseline_sample90_seed130_par_fichier.csv (fuite probable)
reference = "samples/sample90_seed130.rttm"#un fichier .rttm OU un dossier de .rttm
audio_dir = Path("data/voxconverse/dev/audio")
run_name = "pyannote_c1_sample90_seed130"
#B : test complet (D-21 : à n'utiliser que quand on ne règle plus rien) ; voxconverse_test_wav/, pas __MACOSX/ (D-12)
# reference = "data/voxconverse/annotations/test"
# audio_dir = Path("data/voxconverse/test/voxconverse_test_wav")
# run_name = "pyannote_c1_test"

out_dir = Path("results")


def load_reference(source):
    """Renvoie {uri: Annotation} à partir d'un fichier .rttm (sample) ou d'un dossier de .rttm (un par fichier audio)."""
    source = Path(source)
    if source.is_file():
        return load_rttm(str(source))
    annotations = {}
    for rttm in sorted(source.glob("*.rttm")):
        annotations.update(load_rttm(str(rttm)))
    return annotations


def load_pipeline(device):
    """Télécharge (1re fois) puis charge la pipeline. token=None : huggingface_hub lit le token de `hf auth login`."""
    pipeline = Pipeline.from_pretrained(model_id, token=None)
    if pipeline is None:#from_pretrained renvoie None si l'accès est refusé (conditions non acceptées, token absent)
        raise RuntimeError(f"Impossible de charger {model_id} : conditions acceptées ? `hf auth login` fait ?")
    pipeline.to(device)
    return pipeline


def warmup(pipeline, path):
    """Un premier passage sur quelques secondes d'audio : le 1er appel GPU inclut des initialisations (CUDA, allocations)
    qui fausseraient le RTF du premier fichier."""
    audio, file_sr = sf.read(str(path), frames=int(warmup_s * sf.info(str(path)).samplerate),
                             dtype="float32", always_2d=True)#(temps, canaux)
    waveform = torch.from_numpy(audio.T)#pyannote attend (canaux, temps)
    pipeline({"waveform": waveform, "sample_rate": file_sr, "uri": "warmup"})


def diarize_file(pipeline, path, n_ref_speakers=None):
    """La pipeline sur UN fichier. On lui donne le chemin : elle charge l'audio, le passe en 16 kHz mono,
    et prend le nom du fichier (sans .wav) comme uri."""
    kwargs = {"num_speakers": n_ref_speakers} if oracle else {}
    output = pipeline(str(path), **kwargs)
    hyp = output.speaker_diarization#Annotation AVEC overlap (exclusive_speaker_diarization : sans overlap)
    stats_file = {"k_hyp": len(hyp.labels())}
    return hyp, stats_file


def main():
    #0 annotations de référence : HORS de la boucle, donne la liste des fichiers à traiter
    references = load_reference(reference)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pyannote.audio {pyannote.audio.__version__} | torch {torch.__version__} | {model_id} | device {device}")
    pipeline = load_pipeline(device)

    #métriques (D-30 à D-35) : collar 0, UEM [0, durée du wav], agrégation micro (identique à baseline.py)
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)#DER moderne, métrique principale
    der_forgiving = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    det = DetectionErrorRate(collar=0.0, skip_overlap=False)#parole seule : (FA + missed) / parole de référence

    uris = sorted(references)#sorted : ordre déterministe
    if warmup_s > 0 and uris:
        warmup(pipeline, audio_dir / f"{uris[0]}.wav")

    rows = []
    with open(out_dir / f"{run_name}.rttm", "w") as f_rttm:
        for i, uri in enumerate(uris, start=1):
            ref = references[uri]
            path = audio_dir / f"{uri}.wav"
            if not path.exists():
                print(f"[warning] {path} introuvable, fichier ignoré")
                continue

            duration = sf.info(str(path)).duration#D-15
            uem = Timeline([Segment(0, duration)], uri=uri)#D-33

            t0 = time.perf_counter()
            hyp, stats_file = diarize_file(pipeline, path, n_ref_speakers=len(ref.labels()))
            proc_time = time.perf_counter() - t0#chargement compris, évaluation non comprise

            row = {
                "uri": uri,
                "duration_s": duration,
                "n_ref_speakers": len(ref.labels()),
                **stats_file,#k_hyp
                "DER": der(ref, hyp, uem=uem),
                "DER_forgiving": der_forgiving(ref, hyp, uem=uem),
                "JER": jer(ref, hyp, uem=uem),
                "VAD_error": det(ref, hyp.get_timeline().support().to_annotation(), uem=uem),
                "RTF": proc_time / duration,#D-36
            }
            rows.append(row)
            hyp.write_rttm(f_rttm)#hypothèses sauvegardées : on pourra réévaluer sans tout relancer
            print(f"[{i}/{len(uris)}] {uri} | DER {row['DER']:.1%} | k {row['k_hyp']} (réf {row['n_ref_speakers']}) "
                  f"| RTF {row['RTF']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}_par_fichier.csv", index=False)
    total_time = (df["RTF"] * df["duration_s"]).sum()
    print(f"\n=== {run_name} : {len(df)} fichiers, agrégation micro, oracle={oracle} ===")
    print(f"DER moderne   : {abs(der):.2%}")
    print(f"DER forgiving : {abs(der_forgiving):.2%}")
    print(f"JER           : {abs(jer):.2%}")
    print(f"VAD error     : {abs(det):.2%}")
    print(f"RTF global    : {total_time / df['duration_s'].sum():.3f} ({device})")
    print(f"\nDétail du DER moderne :\n{der.report(display=False).tail(1).T}")


if __name__ == "__main__":
    main()