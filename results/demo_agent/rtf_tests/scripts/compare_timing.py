"""Même fichier, même processus : chronométrage façon baseline.py (b.diarize_file, vad_train='devtrain', comme
main()) vs b.diarize_file(vad_train='alldev') vs core.diarize (démo). Appels entrelacés A/B/C, après un 1er appel
de chaque (noté à part). Isole : modèles devtrain vs alldev, surcoût de l'adaptateur / core.diarize.
Usage : python compare_timing.py <csv_out> <n_reps> <fichier1> [...]"""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/grego/projets/Who-Spoke-")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from demo import core  # noqa: E402

engines = core.Engines(load_pyannote=False, log=lambda *a: None)
import baseline as b  # noqa: E402
import vad  # noqa: E402

vad.get_model(train="devtrain")
vad.get_stack_model(train="devtrain")
out, n, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows = []


def A(f, uri):  # comme baseline.main() (n_ref_speakers ignoré car oracle=False)
    t0 = time.perf_counter()
    b.diarize_file(Path(f), uri, n_ref_speakers=1, frame_size=b.frame_size, hop_size=b.hop_size)
    return time.perf_counter() - t0


def B(f, uri):
    t0 = time.perf_counter()
    b.diarize_file(Path(f), uri, frame_size=b.frame_size, hop_size=b.hop_size, vad_train="alldev")
    return time.perf_counter() - t0


def C(f, uri):
    audio = core.load_audio(f)
    _, infos = core.diarize(audio, "baseline", engines)
    return infos["processing_time_s"]


def P(f, uri):  # coût de baseline_adapter.params() seul (appelé dans core.diarize)
    from demo import baseline_adapter
    t0 = time.perf_counter()
    baseline_adapter.params()
    return time.perf_counter() - t0


for f in files:
    uri = Path(f).stem
    dur = b.sf.info(f).duration
    for rep in range(0, n + 1):
        for name, fn in (("A_baselinepy_devtrain", A), ("B_diarize_file_alldev", B), ("C_core_diarize", C),
                         ("P_params", P)):
            t = fn(f, uri)
            rows.append({"file": Path(f).name, "duration_s": dur, "variant": name, "rep": rep, "time_s": t,
                         "RTF": t / dur})
        print(Path(f).name, rep, [round(r["time_s"], 4) for r in rows[-4:]], flush=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
