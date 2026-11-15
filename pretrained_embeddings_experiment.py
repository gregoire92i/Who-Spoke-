"""Expérience : embeddings pré-entraînés WeSpeaker vs statistiques de MFCC, toutes choses égales par ailleurs.

Seule chose qui change : le vecteur de chaque window (baseline.pretrained_embeddings False / True).
Identique : VAD best_vad(train="devtrain") (pas de fuite sur le dev sample 90), MFCC et bornes des windows
(win_segs de compute_embeddings), NME-SC, standardize, windows_to_annotation ; paramètres lus dans baseline.py.
Les deux variantes passent par baseline.diarize_file (le chemin de code de la baseline est celui évalué).

Vérifications intégrées :
  - mêmes bornes de windows dans les deux variantes, pour chaque fichier (colonne same_bounds) ;
  - pas de NaN (compute_pretrained_embeddings lève une erreur sinon) ; --nan_check pqmho : fichier dev sans parole,
    hors sample 90, ni métrique ni CSV, seulement le nombre de windows et leur durée.

Protocole D-30 à D-37 : collar 0, UEM [0, durée du wav], micro ; DER moderne / forgiving / JER + FA, missed, confusion.
Dev uniquement. Un fichier audio à la fois.

Usage : python pretrained_embeddings_experiment.py [--limit N]
        python pretrained_embeddings_experiment.py --nan_check pqmho"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

import baseline as B  # paramètres de la baseline (main() n'est pas exécuté à l'import)
import embeddings as E

OUT_DIR = Path("results/pretrained_emb")
ARMS = {"mfcc_stats": False, "wespeaker": True}  # nom -> valeur de B.pretrained_embeddings

# --- enregistrement des bornes de windows, sans modifier baseline.py ---
# diarize_file appelle les fonctions par le nom importé dans baseline : on les enveloppe le temps de l'expérience
_seen = {}
_orig_compute_embeddings = B.compute_embeddings
_orig_compute_pretrained = B.compute_pretrained_embeddings


def _rec_compute_embeddings(*args, **kwargs):
    X, win_segs = _orig_compute_embeddings(*args, **kwargs)
    _seen["mfcc_win_segs"] = list(win_segs)
    return X, win_segs


def _rec_compute_pretrained(audio, win_segs, **kwargs):
    _seen["pretrained_win_segs"] = list(win_segs)
    return _orig_compute_pretrained(audio, win_segs, **kwargs)


B.compute_embeddings = _rec_compute_embeddings
B.compute_pretrained_embeddings = _rec_compute_pretrained


def warmup():
    """Charge le modèle et fait un passage à vide (initialisations CUDA) : exclu du RTF, comme pyannote_diarization.py."""
    t0 = time.perf_counter()
    model = E.load_pretrained_model()
    dev = next(model.parameters()).device
    with torch.inference_mode():
        model(torch.zeros(2, 1, 2 * B.sr, device=dev))
    return str(dev), time.perf_counter() - t0


def nan_check(uri, audio_dir):
    B.pretrained_embeddings = True
    path = audio_dir / f"{uri}.wav"
    hyp, vad, st = B.diarize_file(path, uri, frame_size=B.frame_size, hop_size=B.hop_size, vad_train="devtrain")
    segs = _seen.get("pretrained_win_segs", [])
    durs = [s.duration for s in segs]
    print(f"{uri} : durée {sf.info(str(path)).duration:.1f} s, parole VAD {vad.duration():.2f} s, "
          f"{st['n_windows']} windows (durées : {sorted(set(round(d, 2) for d in durs))}), k_hyp {st['k_hyp']}, "
          f"pas d'erreur NaN, {len(hyp.labels())} label(s) en sortie")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm", default="samples/dev/sample90_seed130.rttm")
    ap.add_argument("--audio_dir", default="data/voxconverse/dev/audio")
    ap.add_argument("--limit", type=int, default=None, help="n premiers fichiers (test rapide)")
    ap.add_argument("--nan_check", default=None, help="uri dev : vérifie seulement l'absence de NaN")
    args = ap.parse_args()
    if "test" in args.rttm or "test" in args.audio_dir:
        raise SystemExit("Dev uniquement : le test ne sert pas à régler (CLAUDE.md, section 3).")
    audio_dir = Path(args.audio_dir)

    device, t_warm = warmup()
    print(f"Modèle {E.WESPEAKER_ID} chargé sur {device} ({t_warm:.1f} s, exclu du RTF)")
    if device.startswith("cuda"):
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
    if args.nan_check:
        nan_check(args.nan_check, audio_dir)
        return

    tag = Path(args.rttm).stem + (f"_limit{args.limit}" if args.limit else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = {u: a for u, a in load_rttm(args.rttm).items() if (audio_dir / f"{u}.wav").exists()}
    uris = sorted(annotations)[:args.limit] if args.limit else sorted(annotations)

    metrics = {name: dict(der=DiarizationErrorRate(collar=0.0, skip_overlap=False),
                          derf=DiarizationErrorRate(collar=0.0, skip_overlap=True),
                          jer=JaccardErrorRate(collar=0.0, skip_overlap=False)) for name in ARMS}
    det = DetectionErrorRate(collar=0.0, skip_overlap=False)
    rows = {name: [] for name in ARMS}
    rttm_files = {name: open(OUT_DIR / f"{name}_{tag}.rttm", "w") for name in ARMS}

    for i, uri in enumerate(uris, start=1):
        ref = annotations[uri]
        path = audio_dir / f"{uri}.wav"
        duration = sf.info(str(path)).duration
        uem = Timeline([Segment(0, duration)], uri=uri)
        n_ref = len(ref.labels())
        segs_by_arm, line = {}, f"[{i}/{len(uris)}] {uri} (réf {n_ref})"
        for name, flag in ARMS.items():
            B.pretrained_embeddings = flag
            _seen.clear()
            t0 = time.perf_counter()
            hyp, vad, st = B.diarize_file(path, uri, n_ref_speakers=n_ref, frame_size=B.frame_size,
                                          hop_size=B.hop_size, vad_train="devtrain")
            proc = time.perf_counter() - t0
            segs_by_arm[name] = _seen["pretrained_win_segs"] if flag else _seen["mfcc_win_segs"]
            m = metrics[name]
            c = m["der"](ref, hyp, uem=uem, detailed=True)
            rows[name].append({
                "uri": uri, "duration_s": duration, "n_ref_speakers": n_ref, "k_hyp": st["k_hyp"],
                "p_neighbors_used": st["p_neighbors_used"], "n_windows": st["n_windows"],
                "n_short_windows": sum(s.duration < B.win_s - 1e-6 for s in segs_by_arm[name]),
                "DER": c["diarization error rate"], "DER_forgiving": m["derf"](ref, hyp, uem=uem),
                "JER": m["jer"](ref, hyp, uem=uem), "VAD_error": det(ref, vad.to_annotation(), uem=uem) if not flag else np.nan,
                "FA_s": c["false alarm"], "missed_s": c["missed detection"], "confusion_s": c["confusion"],
                "total_s": c["total"], "RTF": proc / duration, "device": device if flag else "cpu",
            })
            hyp.write_rttm(rttm_files[name])
            line += f" | {name}: DER {rows[name][-1]['DER']:.1%} k {st['k_hyp']}"
        same = segs_by_arm["mfcc_stats"] == segs_by_arm["wespeaker"]
        for name in ARMS:
            rows[name][-1]["same_bounds"] = same
        print(line + f" | mêmes bornes : {same}", flush=True)

    for f in rttm_files.values():
        f.close()

    agg = []
    for name in ARMS:
        df = pd.DataFrame(rows[name])
        if name == "wespeaker":  # la VAD est la même : son erreur n'est comptée qu'une fois (colonne mfcc_stats)
            df["VAD_error"] = pd.DataFrame(rows["mfcc_stats"])["VAD_error"]
        df.to_csv(OUT_DIR / f"{name}_{tag}_par_fichier.csv", index=False)
        m, tot = metrics[name], df["total_s"].sum()
        agg.append({
            "variante": name, "n_fichiers": len(df), "device_embeddings": df["device"].iloc[0],
            "DER_%": 100 * abs(m["der"]), "DER_forgiving_%": 100 * abs(m["derf"]), "JER_%": 100 * abs(m["jer"]),
            "FA_%": 100 * df["FA_s"].sum() / tot, "missed_%": 100 * df["missed_s"].sum() / tot,
            "confusion_%": 100 * df["confusion_s"].sum() / tot, "VAD_error_%": 100 * abs(det),
            "RTF": (df["RTF"] * df["duration_s"]).sum() / df["duration_s"].sum(),
            "k_exact_%": 100 * (df["k_hyp"] == df["n_ref_speakers"]).mean(),
            "k_sous_estime_%": 100 * (df["k_hyp"] < df["n_ref_speakers"]).mean(),
            "k_sur_estime_%": 100 * (df["k_hyp"] > df["n_ref_speakers"]).mean(),
            "abs_k_err_moyen": (df["k_hyp"] - df["n_ref_speakers"]).abs().mean(),
            "fichiers_memes_bornes": int(df["same_bounds"].sum()),
        })
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT_DIR / f"agregats_{tag}.csv", index=False)
    print(f"\n=== Embeddings MFCC vs WeSpeaker, {tag}, VAD best_vad(train='devtrain'), agrégation micro ===")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
