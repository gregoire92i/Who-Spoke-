# Who Spoke? — Qui parle, et quand ?

Système de diarisation de locuteurs : identifier les différents locuteurs d'un enregistrement audio et déterminer leurs temps de parole. Le projet va de la recherche de corpus à l'exploration de modèles de deep learning, en passant par le développement de notre propre pipeline de diarisation, la comparaison avec les technologies existantes et une démonstration fonctionnelle.

Corpus principal : VoxConverse, annotations v0.3 (voir [`docs/decision_corpus.md`](docs/decision_corpus.md)).

## Arborescence

```
Who-Spoke-/
├── README.md
├── requirements.txt
├── .gitignore
├── experimenations.py            # exploration audio : forme d'onde, spectrogramme, énergie
├── plot_reference.py             # timeline de référence d'un fichier RTTM
├── rttm_files.py                 # statistiques par fichier, tirage du sous-ensemble de dev
├── Data_vizualizer.py            # statistiques du corpus, comparaison sous-ensemble / dev complet
├── metrics_and_sanity_checks.py  # sanity checks de la chaîne d'évaluation
├── figures/                      # figures générées par les scripts
├── data/                         # non versionné
└── docs/
    ├── corpus/
    │   ├── voxconverse.md
    │   └── ami.md
    ├── decision_corpus.md
    ├── choix_techniques.md
    ├── utilisation_claude.md
    └── notes-importantes-et-difficultes.md
```

## Prérequis

- Linux (développé sous Ubuntu via WSL2)
- Python 3.14.4
- Paquets système : `python3-venv`, `ffmpeg`, `libsndfile1`, `unzip`
- Testé avec un GPU NVIDIA (pilote compatible CUDA 13.2, `torch 2.15.0+cu132`)

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

## Protocole d'évaluation

Le protocole (métrique, collar, traitement des overlaps, UEM, agrégation) est décrit dans [`docs/choix_techniques.md`](docs/choix_techniques.md).