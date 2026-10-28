# "Who Spoke?"
## un système d’intelligence artificielle capable d’identifier les différents locuteurs d’un enregistrement audio et de déterminer leurs temps de parole. De la recherche de datasets à l’exploration des modèles de deep learning, en devellopant notre propre pipeline de diarisation, compare les technologies existantes et réalise une démonstration fonctionnelle.

Corpus principal : VoxConverse, annotations v0.3 (voir [`docs/decision_corpus.md`](docs/decision_corpus.md)).

## Arborescence

```
Who-Spoke-/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                       # non versionné
└── docs/
    ├── corpus/
    │   ├── voxconverse.md
    │   └── ami.md
    ├── decision_corpus.md
    └── notes-importantes-et-difficultes.md
```

## Prérequis

- Linux (développé sous Ubuntu via WSL2)
- Python 3.14.4
- Paquets système : `python3-venv`, `ffmpeg`, `libsndfile1`

## Installation

```bash
git clone https://github.com/gregoire92i/Who-Spoke-.git
cd Who-Spoke-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Non redistribuées (droits d'auteur des vidéos). À télécharger dans `data/` (exclu par `.gitignore`) :

```bash
mkdir -p data/voxconverse && cd data/voxconverse
wget -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
wget -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
unzip -q voxconverse_dev_wav.zip  -d dev
unzip -q voxconverse_test_wav.zip -d test
git clone --depth 1 https://github.com/joonson/voxconverse.git annotations   # RTTM v0.3
```
