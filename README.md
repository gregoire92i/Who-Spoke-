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

Audio : page de téléchargement hébergée par le laboratoire de J. S. Chung (KAIST), <https://mm.kaist.ac.kr/datasets/voxconverse/>.
Annotations RTTM v0.3 : <https://github.com/joonson/voxconverse> (branche `master`).

À executer dans le dossier de votre projet pour telecharger directement le dataset :

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
