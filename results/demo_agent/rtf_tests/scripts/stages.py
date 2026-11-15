"""Décomposition par étape de baseline.diarize_file (vad_train="alldev", comme la démo), sans modifier les modules :
mêmes fonctions, mêmes paramètres (lus dans baseline.py), même ordre ; chaque appel est chronométré.
best_vad est découpé en ses deux appels internes (vad.compute_features puis vad.vad_stack avec PROD_PARAMS).
Contrôle : la sortie est comparée à celle de baseline.diarize_file sur le même fichier.
Usage : python stages.py <csv_out> <n_calls> <fichier1> [...]  (un processus neuf, modèles chargés comme la démo)."""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/grego/projets/Who-Spoke-")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from demo import core  # noqa: E402,F401  (fixe les variables d'environnement comme la démo)
from demo import baseline_adapter  # noqa: E402

ok, msg = baseline_adapter.prepare(log=lambda *a: None)
assert ok, msg
import baseline as b  # noqa: E402
import vad  # noqa: E402

out, n, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
P = {**vad.PROD_PARAMS, "train": "alldev"}


def run(path, uri):
    t = {}
    c = time.perf_counter
    s = c(); audio = b.load_audio(str(path), b.sr); t["1_load_audio"] = c() - s
    s = c(); D = b.FFT(audio, frame_size=b.frame_size, hop_size=b.hop_size); t["2_stft"] = c() - s
    s = c(); F, _ = vad.compute_features(audio); t["3a_vad_features"] = c() - s
    s = c(); vtl = vad.vad_stack(F, uri=uri, **P); t["3b_vad_stack"] = c() - s
    del audio, F
    s = c(); mfcc = b.compute_mfcc(D, sr=b.sr, n_mfcc=b.n_mfcc); t["4_mfcc"] = c() - s
    del D
    s = c()
    X, win_segs = b.compute_embeddings(mfcc, vtl, win_s=b.win_s, hop_s=b.hop_s, min_s=b.min_s, stats=b.stats,
                                       drop_c0=b.drop_c0, mode=b.mode, sr=b.sr, hop_size=b.hop_size)
    t["5_embeddings"] = c() - s
    s = c()
    assert b.clustering_method == "nme"
    labels, info = b.nme_spectral_clustering(X, oracle=b.oracle, n_speakers=None, min_speakers=b.min_speakers,
                                             max_speakers=b.max_speakers, min_neighbors=b.min_neighbors,
                                             max_frac=b.nme_max_frac, n_grid=b.nme_n_grid,
                                             laplacian=b.nme_laplacian, standardize=b.standardize, seed=b.seed,
                                             return_info=True)
    t["6_clustering"] = c() - s
    s = c()
    hyp, post = b.windows_to_annotation(labels, X, win_segs, vtl, n_frames=mfcc.shape[1], uri=uri, sr=b.sr,
                                        hop_size=b.hop_size, return_info=True)
    t["7_post_traitement"] = c() - s
    return hyp, t, len(X)


rows = []
for f in files:
    uri = Path(f).stem
    for rep in range(1, n + 1):
        s = time.perf_counter()
        hyp, t, nw = run(f, uri)
        tot = time.perf_counter() - s
        rows.append({"file": Path(f).name, "rep": rep, "duration_s": b.sf.info(f).duration, "n_windows": nw,
                     **t, "sum_s": sum(t.values()), "total_s": tot})
        print(Path(f).name, rep, {k: round(v, 3) for k, v in t.items()}, flush=True)
    ref, _, _ = b.diarize_file(Path(f), uri, frame_size=b.frame_size, hop_size=b.hop_size, vad_train="alldev")
    same = list(ref.itertracks(yield_label=True)) == list(hyp.itertracks(yield_label=True))
    print("identique à baseline.diarize_file :", same, flush=True)
    for r in rows:
        if r["file"] == Path(f).name:
            r["identique_diarize_file"] = same
new = not Path(out).exists()
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
