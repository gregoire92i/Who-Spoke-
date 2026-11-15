"""Baseline dans la démo : appelle baseline.diarize_file tel quel, avec vad_train="alldev" (modèles de production
de best_vad, entraînés sur les 216 fichiers dev ; cohérent pour des enregistrements inconnus).
Toutes les autres valeurs (frames, MFCC, windows, NME-SC, max_speakers, seed…) sont lues dans baseline.py :
aucun paramètre n'est recopié ici."""
from pathlib import Path


from demo import ROOT

VAD_TRAIN = "alldev"
# modèles de production de best_vad (noms construits comme dans vad.get_model / vad.get_stack_model)
MODELS_DIR = ROOT / "results" / "vad_agent" / "models"
MODEL_FILES = ("hgb_base+local_s4_trainalldev.pkl", "stack_hgb_base+local_s4_alldev.pkl")
RETRAIN_CMD = "python vad.py cache --split dev --rttm samples/dev/all_dev.rttm"


def check_vad_models():
    """État des modèles best_vad alldev. Renvoie (état, message) avec état parmi :
      "ok" : les 2 modèles sont présents ;
      "retrain" : un modèle manque mais les prérequis du réentraînement sont là (vad.py le fera tout seul) ;
      "missing" : un modèle manque et les prérequis aussi -> baseline indisponible."""
    missing = [f for f in MODEL_FILES if not (MODELS_DIR / f).exists()]
    if not missing:
        return "ok", "modèles best_vad (alldev) présents"
    all_dev = ROOT / "samples" / "dev" / "all_dev.rttm"
    problems = []
    if not all_dev.exists():
        problems.append("samples/dev/all_dev.rttm absent")
    else:
        from pyannote.database.util import load_rttm
        uris = sorted(load_rttm(str(all_dev)))
        n_cache = sum((ROOT / "results/vad_agent/cache/dev" / f"{u}.npz").exists() for u in uris)
        n_ann = sum((ROOT / "data/voxconverse/annotations/dev" / f"{u}.rttm").exists() for u in uris)
        if n_cache < len(uris):
            problems.append(f"cache de caractéristiques incomplet ({n_cache}/{len(uris)} fichiers dans "
                            f"results/vad_agent/cache/dev/)")
        if n_ann < len(uris):
            problems.append(f"annotations dev incomplètes ({n_ann}/{len(uris)} dans data/voxconverse/annotations/dev/)")
    if problems:
        return "missing", (f"modèles best_vad absents ({', '.join(missing)}) et réentraînement impossible : "
                           f"{' ; '.join(problems)}. Pour régénérer : télécharger VoxConverse dev (README), puis "
                           f"`{RETRAIN_CMD}` (~10 min), puis relancer la démo (réentraînement ~3-5 min).")
    return "retrain", (f"modèles best_vad absents ({', '.join(missing)}) : réentraînement automatique en cours "
                       f"(environ 3 à 5 min, une seule fois)")


def prepare(log=print):
    """Vérifie les modèles best_vad (alldev) et les charge une seule fois en mémoire (vad._MODELS), sans aucun
    calcul sur un signal. S'ils manquent et que les prérequis sont là, vad.py les réentraîne ici, au démarrage.
    Renvoie (disponible, message)."""
    state, msg = check_vad_models()
    if state == "missing":
        return False, msg
    if state == "retrain":
        log(f"[baseline] {msg}")
    import vad
    try:
        vad.get_model(train=VAD_TRAIN)        # 1er étage (hgb_base+local_s4_trainalldev.pkl)
        vad.get_stack_model(train=VAD_TRAIN)  # 2e étage (stack_hgb_base+local_s4_alldev.pkl)
    except Exception as e:
        return False, f"chargement / réentraînement de best_vad impossible : {type(e).__name__}: {e}"
    return True, ("baseline prête (réentraînement terminé)" if state == "retrain" else "baseline prête") + \
        f" ; VAD best_vad(train='{VAD_TRAIN}')"


def params():
    """Paramètres effectivement utilisés, lus dans baseline.py et vad.py (affichage)."""
    import baseline as b
    import vad
    return {"VAD": f"best_vad(train='{VAD_TRAIN}')",
            "VAD params": {k: v for k, v in vad.PROD_PARAMS.items() if k != "train"},
            "frames / hop (ms)": f"{b.frame_ms} / {b.hop_ms}", "n_mfcc": b.n_mfcc,
            "windows (s)": f"win {b.win_s}, hop {b.hop_s}, min {b.min_s}", "stats": b.stats, "mode": b.mode,
            "clustering": b.clustering_method,
            **({"laplacian": b.nme_laplacian, "nme_max_frac": b.nme_max_frac, "nme_n_grid": b.nme_n_grid}
               if b.clustering_method == "nme" else {"p_neighbors": b.p_neighbors}),
            "min_neighbors": b.min_neighbors, "speakers": f"{b.min_speakers} à {b.max_speakers} (eigengap)",
            "standardize": b.standardize, "seed": b.seed}


def diarize(path, uri, vad_train=VAD_TRAIN):
    """Un fichier -> (Annotation, paramètres, infos supplémentaires). vad_train n'est changé que par le test
    d'égalité avec baseline.diarize_file (tests/test_demo.py)."""
    import baseline as b
    hyp, vad_tl, stats_file = b.diarize_file(Path(path), uri, frame_size=b.frame_size, hop_size=b.hop_size,
                                             vad_train=vad_train)
    p = params()
    p["VAD"] = f"best_vad(train='{vad_train}')"
    extra = {"vad_speech_s": vad_tl.duration(), **{k: stats_file[k] for k in ("p_neighbors_used", "n_windows")}}
    return hyp, p, extra
