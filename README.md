# Who Spoke? — Qui parle, et quand ?

Système de diarisation de locuteurs : identifier les différents locuteurs d'un enregistrement audio et déterminer leurs temps de parole. Le projet va de la recherche de corpus à l'exploration de modèles de deep learning, en passant par le développement de notre propre pipeline de diarisation, la comparaison avec les technologies existantes et une démonstration fonctionnelle.

Corpus principal : VoxConverse, annotations v0.3 (voir [`docs/decision_corpus.md`](docs/decision_corpus.md)).

## Arborescence

```
Who-Spoke-/
├── README.md
├── requirements.txt
├── .gitignore
├── CLAUDE.md                     # règles de travail de Claude Code dans le dépôt
├── baseline.py                   # point d'entrée : baseline complète sur un sample + évaluation
├── baseline_oracle.py            # diagnostics oracle (VAD, nombre de locuteurs, vote)
├── ablation.py                   # ablation des 2 améliorations (test sample 90, version figée)
├── nme_sc_experiment.py          # expérience NME-SC (choix de p par fichier) sur le dev
├── pyannote_diarization.py       # diarisation avec la pipeline pré-entraînée pyannote
├── audio_features.py             # chargement audio, STFT, VAD par énergie, MFCC
├── vad.py                        # VAD apprise (vad_stack / best_vad) et évaluation de la VAD seule
├── embeddings.py                 # embeddings par fenêtre glissante (statistiques des MFCC)
├── clustering.py                 # clustering spectral (p fixe ou NME-SC), nombre de locuteurs par eigengap
├── post_treatment.py             # fenêtres -> trames -> Annotation pyannote
├── baseline_embeddings_tests.py  # balayage de la taille des windows d'embedding
├── setting_frames_nd_hop.py      # balayage frame / hop de la STFT
├── rttm_files.py                 # statistiques par fichier, tirage des samples
├── plot_reference.py             # timeline de référence d'un fichier RTTM
├── Data_vizualizer.py            # statistiques du corpus, comparaison sample / dev complet
├── metrics_and_sanity_checks.py  # sanity checks de la chaîne d'évaluation
├── demo/                         # démo Gradio (voir « Démo »)
├── tests/test_demo.py            # tests de la démo (pytest)
├── analysis/clustering_agent/    # scripts du diagnostic « k oracle vs eigengap »
├── samples/
│   ├── dev/                      # sample90_seed130, sample40_seed130, all_dev (RTTM)
│   └── test/                     # sample90_seed130, sample40_seed130 (RTTM)
├── results/                      # sorties des scripts (RTTM des hypothèses, CSV par fichier, rapports)
├── figures/                      # figures générées par les scripts
├── data/                         # non versionné
└── docs/
    ├── corpus/
    │   ├── voxconverse.md
    │   └── ami.md
    ├── sync/                     # rapports de synchronisation
    ├── enonce_projet.md
    ├── etat_du_projet.md
    ├── decision_corpus.md
    ├── choix_techniques.md
    ├── rapport-ablation.md
    ├── utilisation_claude.md
    └── notes-importantes-et-difficultes.md
```

## Prérequis

- Linux (développé sous Ubuntu via WSL2)
- Python 3.14.4
- Paquets système : `python3-venv`, `ffmpeg`, `libsndfile1`, `unzip`
- Testé avec un GPU NVIDIA (pilote compatible CUDA 13.2, `torch 2.15.0+cu132`). La baseline et la démo tournent aussi sur CPU.

## Installation

```bash
git clone https://github.com/gregoire92i/Who-Spoke-.git
cd Who-Spoke-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

VoxConverse est distribué sous licence CC BY 4.0. Si vous l'utilisez, citez :

> J. S. Chung, J. Huh, A. Nagrani, T. Afouras, A. Zisserman. *Spot the conversation: speaker diarisation in the wild*. Interspeech, 2020. [arXiv:2007.01216](https://arxiv.org/abs/2007.01216)

```bibtex
@inproceedings{chung2020spot,
  title     = {Spot the conversation: speaker diarisation in the wild},
  author    = {Chung, Joon Son and Huh, Jaesung and Nagrani, Arsha and Afouras, Triantafyllos and Zisserman, Andrew},
  booktitle = {Interspeech},
  year      = {2020}
}
```

L'audio n'est pas redistribué dans ce dépôt (droits d'auteur des vidéos). Il doit être téléchargé dans `data/`, exclu par `.gitignore`.

- Audio : page de téléchargement hébergée par le laboratoire de J. S. Chung (KAIST), <https://mm.kaist.ac.kr/datasets/voxconverse/>.
- Annotations RTTM v0.3 : <https://github.com/joonson/voxconverse> (branche `master`).

À exécuter à la racine du dépôt pour télécharger le corpus :

```bash
mkdir -p data/voxconverse && cd data/voxconverse
wget -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip
wget -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip
md5sum -c <<EOF
2a6e07e7473d9841abb132554a698a36  voxconverse_dev_wav.zip
834558bbd9b1ffd2d4893181556ceddd  voxconverse_test_wav.zip
EOF
unzip -q voxconverse_dev_wav.zip  -d dev
unzip -q voxconverse_test_wav.zip -d test
git clone --depth 1 https://github.com/joonson/voxconverse.git annotations
```

Le serveur coupe parfois le téléchargement : relancer la même commande `wget -c` reprend là où il s'est arrêté.

Structure obtenue :

```
data/voxconverse/
├── dev/audio/                     # 216 fichiers .wav
├── test/voxconverse_test_wav/     # 232 fichiers .wav
├── test/__MACOSX/                 # métadonnées macOS (faux ._*.wav), à ignorer
└── annotations/
    ├── dev/                       # RTTM v0.3
    └── test/
```

## Lancer la baseline

Tous les scripts utilisent des chemins relatifs : ils doivent être lancés **depuis la racine du dépôt**, avec le `.venv` activé.

1. Sample de dev : `baseline.py` lit la liste des fichiers à traiter dans `samples/dev/sample90_seed130.rttm` (90 fichiers du dev, tirage stratifié, seed 130). Il est possible de régénérer un sample avec le code en commentaire à la fin de `rttm_files.py`. Le sample sera le même si la seed n'est pas modifiée.

2. Lancer la baseline (VAD apprise `best_vad` → MFCC → embeddings → clustering spectral NME-SC → post-traitement → évaluation) :

```bash
python baseline.py
```

Les paramètres (VAD, MFCC, fenêtres, clustering, sample utilisé) sont regroupés en haut de `baseline.py`, où leurs valeurs font foi. Les fichiers sont traités un par un pour limiter la RAM.

La VAD `best_vad(train="devtrain")` utilise des modèles entraînés **hors** du sample 90 de dev (pas de fuite) ; ils sont versionnés dans `results/vad_agent/models/`. Le rapport de la VAD est dans `results/vad_agent/rapport.md`.

Sorties dans `results/` :

- `baseline_sample90_seed130.rttm` : hypothèses de diarisation de tous les fichiers ;
- `baseline_sample90_seed130_par_fichier.csv` : scores par fichier (DER, DER forgiving, JER, erreur de VAD, RTF, nombre de locuteurs estimé et de référence).

Les scores agrégés (micro) sont affichés dans le terminal.

## Autres scripts

Même règle : depuis la racine du dépôt, `.venv` activé. Les paramètres sont en haut de chaque script.

| Commande | Rôle | Fichiers | Sorties |
|---|---|---|---|
| `python baseline_oracle.py` | remplace un composant par la vérité terrain (`oracle_vad`, `oracle_k`, `oracle_vote`, un seul à la fois) | dev sample 90 | `results/baseline_oracle_*` |
| `python nme_sc_experiment.py [--limit N]` | compare p fixe et NME-SC (Laplacien normalisé / non normalisé) sur les mêmes embeddings | dev sample 90 | `results/nme_sc/` |
| `python ablation.py [--limit N]` | ablation des 2 améliorations (VAD, clustering), 4 configurations, version figée avant exécution (D-62) | **test** sample 90 | `results/ablation/`, rapport : [`docs/rapport-ablation.md`](docs/rapport-ablation.md) |
| `python pyannote_diarization.py` | pipeline pré-entraînée `pyannote/speaker-diarization-community-1`, même protocole | test sample 90 | `results/pyannote/` |
| `python vad.py eval --method … --rttm …` | évaluation de la VAD seule (detection error) | dev | `results/vad_agent/` |

Accès au modèle pyannote (une seule fois) : accepter les conditions sur <https://huggingface.co/pyannote/speaker-diarization-community-1>, créer un token en lecture, puis `hf auth login`. Le token est stocké dans `~/.cache/huggingface`, jamais dans le code.

Règle du projet : le dev sert à régler, le test à mesurer. Le test n'est jamais utilisé pour choisir un paramètre.

## Démo

Application Gradio locale : charger un audio (ou enregistrer au micro), lancer la diarisation (baseline ou pyannote), afficher la timeline des locuteurs et leurs temps de parole, écouter les segments, renommer les locuteurs, exporter en RTTM. Si un RTTM de référence est fourni, le DER et le JER sont affichés.

```bash
source .venv/bin/activate
python -m demo.app
```

Puis ouvrir <http://127.0.0.1:7860>. La démo écoute uniquement en local (pas de lien public) et fonctionne sans internet une fois les modèles en cache.

- `WHO_SPOKE_PORT=7870` : autre port ;
- `WHO_SPOKE_DEVICE=cpu` : forcer pyannote sur CPU.

Formats acceptés : wav et flac directement, les autres (mp3, m4a…) sont convertis par `ffmpeg`. Les fichiers temporaires vont dans `demo/tmp/` (non versionné). La baseline de la démo utilise `best_vad(train="alldev")` (modèles entraînés sur tout le dev). Architecture et rapport : `results/demo_agent/`.

## Tests

```bash
python -m pytest tests/
```

## Résultats principaux

Protocole : collar 0, UEM = [0, durée du wav], agrégation micro. Ablation sur le **test sample 90** ([`docs/rapport-ablation.md`](docs/rapport-ablation.md)) :

| Configuration | DER moderne | DER forgiving | JER |
|---|---|---|---|
| `energy_vad` + p fixe 0.1 (départ) | 44,31 % | 43,16 % | 77,19 % |
| `best_vad` + p fixe 0.1 (amélioration n°1) | 36,89 % | 35,40 % | 77,00 % |
| `energy_vad` + NME-SC (amélioration n°2) | 38,38 % | 36,73 % | 68,47 % |
| `best_vad` + NME-SC (les deux) | 30,86 % | 28,77 % | 68,38 % |

Plancher « un seul locuteur » sur le même sample : DER moderne 56,7 %. Les autres résultats (dev, oracles, pyannote) sont dans [`docs/etat_du_projet.md`](docs/etat_du_projet.md) et `results/`.

## Documentation

- [`docs/etat_du_projet.md`](docs/etat_du_projet.md) : avancement, pipeline actuel, résultats de référence.
- [`docs/choix_techniques.md`](docs/choix_techniques.md) : toutes les décisions (D-xx), dont le protocole d'évaluation (métrique, collar, overlaps, UEM, agrégation).
- [`docs/rapport-ablation.md`](docs/rapport-ablation.md) : ablation des 2 améliorations.
- [`docs/corpus/`](docs/corpus/) et [`docs/decision_corpus.md`](docs/decision_corpus.md) : fiches des corpus et choix de VoxConverse.
- [`docs/notes-importantes-et-difficultes.md`](docs/notes-importantes-et-difficultes.md) : difficultés rencontrées.
- [`docs/utilisation_claude.md`](docs/utilisation_claude.md) : usage de Claude et provenance des fichiers.
