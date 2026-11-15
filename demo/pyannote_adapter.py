"""pyannote dans la démo : réutilise load_pipeline et diarize_file de pyannote_diarization.py.
HF_HUB_OFFLINE=1 est fixé dans demo/__init__.py avant tout import : le modèle est lu dans le cache local
(~/.cache/huggingface/hub), sans réseau et sans jeton. Aucun jeton n'est lu ni affiché par la démo.
Appareil : GPU si disponible, sinon CPU ; WHO_SPOKE_DEVICE=cpu force le CPU (plan B)."""
import os


def load(log=print):
    """Charge la pipeline une seule fois. Renvoie (pipeline, device)."""
    import torch
    import pyannote_diarization as pd_mod
    forced = os.environ.get("WHO_SPOKE_DEVICE", "").lower()
    if forced == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    pipeline = pd_mod.load_pipeline(device)
    return pipeline, device


def params():
    import pyannote_diarization as pd_mod
    import pyannote.audio
    return {"modèle": pd_mod.model_id, "pyannote.audio": pyannote.audio.__version__,
            "nombre de locuteurs": "estimé par la pipeline (oracle=False)",
            "sortie": "speaker_diarization (overlap compris)"}


def diarize(pipeline, path):
    """Un fichier -> (Annotation, paramètres, infos supplémentaires)."""
    import pyannote_diarization as pd_mod
    hyp, _ = pd_mod.diarize_file(pipeline, path)  # n_ref_speakers=None, oracle=False dans pyannote_diarization.py
    return hyp, params(), {}
