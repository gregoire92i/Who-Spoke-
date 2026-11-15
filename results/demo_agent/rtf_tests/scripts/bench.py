"""Banc RTF de la démo (sans chauffe). Un processus neuf par appel de ce script, comme un lancement de l'interface.
Usage : python bench.py <csv_out> <tag> <pipeline> <n_calls> <fichier1> [fichier2 ...]
Chaque fichier est analysé n_calls fois de suite (load_audio puis core.diarize), dans l'ordre donné.
Les deux pipelines sont chargées au démarrage, comme demo.app.main()."""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/grego/projets/Who-Spoke-")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
t_proc0 = time.perf_counter()
from demo import core  # noqa: E402

out, tag, pipeline, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
files = sys.argv[5:]
t0 = time.perf_counter()
engines = core.Engines(log=lambda *a: None)
startup = time.perf_counter() - t0
import_s = t0 - t_proc0
assert pipeline in engines.available(), engines.status
rows, call = [], 0
for f in files:
    for rep in range(1, n + 1):
        call += 1
        a = time.perf_counter()
        audio = core.load_audio(f)
        b = time.perf_counter()
        ann, infos = core.diarize(audio, pipeline, engines)
        c = time.perf_counter()
        if audio["converted"]:
            Path(audio["pipeline_path"]).unlink(missing_ok=True)  # wav converti par la démo : supprimé
        rows.append(dict(tag=tag, file=Path(f).name, format=Path(f).suffix[1:], pipeline=pipeline, rep=rep,
                         call_index=call, import_s=round(import_s, 3), startup_s=round(startup, 3),
                         load_s=b - a, diarize_s=infos["processing_time_s"], total_s=c - a,
                         duration_s=infos["duration_s"], RTF_affiche=infos["RTF"],
                         RTF_total=(c - a) / infos["duration_s"], converted=audio["converted"],
                         k_hyp=infos["k_hyp"], device=str(engines.device)))
        print(f"{tag} {Path(f).name} {pipeline} rep{rep} load {b-a:.3f} diar {infos['processing_time_s']:.3f} "
              f"RTF {infos['RTF']:.4f}", flush=True)
new = not Path(out).exists()
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
