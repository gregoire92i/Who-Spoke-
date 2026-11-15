"""Démo Who Spoke? : charger un audio -> diarisation -> timeline, temps de parole, écoute, export RTTM.

Ce fichier est exécuté AVANT tout autre import du paquet : il fixe les variables d'environnement qui doivent
l'être avant l'import de huggingface_hub, pyannote, gradio et matplotlib.
- HF_HUB_OFFLINE=1 : pyannote est chargé depuis le cache local, sans réseau et sans jeton (D-05).
- GRADIO_ANALYTICS_ENABLED=False : aucune statistique d'usage envoyée par Gradio.
- GRADIO_TEMP_DIR : les fichiers déposés dans l'interface restent dans demo/tmp/ (ignoré par git).
- MPLBACKEND=Agg : pas d'affichage matplotlib (D-06)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # racine du dépôt
TMP_DIR = ROOT / "demo" / "tmp"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", str(TMP_DIR / "gradio"))
os.environ.setdefault("MPLBACKEND", "Agg")
