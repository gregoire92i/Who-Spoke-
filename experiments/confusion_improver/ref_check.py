"""Tour 0, étape 1 : reproduction de la référence (baseline.py, NME-SC normalisé, best_vad devtrain, MFCC)
sur le dev sample 90, SANS modifier baseline.py : le drapeau baseline.pretrained_embeddings est forcé à False
en mémoire (dans ce processus uniquement), puis baseline.diarize_file est appelé tel quel.
Rien n'est écrit hors de results/confusion_improver/. Dev uniquement.
Usage (depuis la racine du dépôt) : python experiments/confusion_improver/ref_check.py"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

import baseline as B

RTTM = ROOT / "samples/dev/sample90_seed130.rttm"
AUDIO = ROOT / "data/voxconverse/dev/audio"
OUT = ROOT / "results/confusion_improver"


def main():
    print(f"baseline.pretrained_embeddings lu dans baseline.py : {B.pretrained_embeddings}")
    B.pretrained_embeddings = False  # forcé en mémoire, fichier inchangé
    print(f"clustering_method={B.clustering_method} nme_laplacian={B.nme_laplacian} max_speakers={B.max_speakers} "
          f"win_s={B.win_s} hop_s={B.hop_s} seed={B.seed}")
    anns = load_rttm(str(RTTM))
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    derf = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    rows = []
    for i, uri in enumerate(sorted(anns), 1):
        ref = anns[uri]
        path = AUDIO / f"{uri}.wav"
        dur = sf.info(str(path)).duration
        uem = Timeline([Segment(0, dur)], uri=uri)
        t0 = time.perf_counter()
        hyp, vad, st = B.diarize_file(path, uri, n_ref_speakers=len(ref.labels()), frame_size=B.frame_size,
                                      hop_size=B.hop_size, vad_train="devtrain")
        proc = time.perf_counter() - t0
        c = der(ref, hyp, uem=uem, detailed=True)
        rows.append({"uri": uri, "duration_s": dur, "n_ref_speakers": len(ref.labels()), "k_hyp": st["k_hyp"],
                     "DER": c["diarization error rate"], "DER_forgiving": derf(ref, hyp, uem=uem),
                     "JER": jer(ref, hyp, uem=uem), "FA_s": c["false alarm"], "missed_s": c["missed detection"],
                     "confusion_s": c["confusion"], "total_s": c["total"], "RTF": proc / dur,
                     "n_speech_frames": st["n_speech_frames"], "n_uncovered": st["n_uncovered"], "n_tie": st["n_tie"]})
        print(f"[{i}/{len(anns)}] {uri} DER {rows[-1]['DER']:.1%} k {st['k_hyp']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ref_check_sample90_par_fichier.csv", index=False)
    tot = df["total_s"].sum()
    print(f"\nDER moderne {abs(der):.4%} | forgiving {abs(derf):.4%} | JER {abs(jer):.4%}")
    print(f"FA {df['FA_s'].sum()/tot:.4%} | missed {df['missed_s'].sum()/tot:.4%} | confusion {df['confusion_s'].sum()/tot:.4%}")
    print(f"RTF {(df['RTF']*df['duration_s']).sum()/df['duration_s'].sum():.4f}")


if __name__ == "__main__":
    main()
