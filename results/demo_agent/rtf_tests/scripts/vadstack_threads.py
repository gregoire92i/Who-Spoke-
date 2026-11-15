"""Microbanc : temps de vad.compute_features et vad.vad_stack (modèles alldev) répétés 30 fois sur le même signal.
Lancé avec et sans OMP_NUM_THREADS=1 (variable d'environnement, aucun code modifié)."""
import os, sys, time, statistics as st
sys.path.insert(0, "/home/grego/projets/Who-Spoke-"); os.chdir("/home/grego/projets/Who-Spoke-")
from demo import core  # noqa
import vad, librosa
P = {**vad.PROD_PARAMS}
vad.get_model(train="alldev"); vad.get_stack_model(train="alldev")
for f in sys.argv[1:]:
    y, _ = librosa.load(f, sr=16000)
    tf, ts = [], []
    for i in range(31):
        a = time.perf_counter(); F, _ = vad.compute_features(y); b = time.perf_counter()
        vad.vad_stack(F, **P); c = time.perf_counter()
        if i: tf.append(b - a); ts.append(c - b)
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','(défaut)')} {os.path.basename(f)} "
          f"features med {st.median(tf):.4f} min {min(tf):.4f} max {max(tf):.4f} | vad_stack med {st.median(ts):.4f} "
          f"min {min(ts):.4f} max {max(ts):.4f} | vad_stack: {' '.join(f'{v:.3f}' for v in ts)}", flush=True)
