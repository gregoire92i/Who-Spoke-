# Utilisation de Claude dans le projet Who Spoke ?

> Ce document explique comment l'assistant IA Claude (Anthropic) a été utilisé dans ce projet, et ce qui vient de Grégoire ou de Claude. Dernière mise à jour : 04/11/2026.

## 1. Outil

- **Assistant** : Claude, développé par Anthropic, utilisé via claude.ai dans un « Projet » dédié qui partage les fichiers de contexte entre les conversations.
- **Modèle(s)** : *O**p**u**s** **5.5** **et **S**o**nnet** **5**.**5** *(le modèle peut changer d'une conversation à l'autre).
- **Données transmises à Claude** :
  - des extraits de code ;
  - des sorties de terminal ;
  - des fichiers de documentation du projet.
  - **Aucun fichier audio du corpus** ne lui a été transmis.

## 2. Règles d'usage

Ces règles ont été fixées au début du projet, dans les instructions de celui-ci :

1. **Claude est un mentor, pas un générateur de solutions.** Il explique, compare et corrige. Il ne génère jamais de code sans instruction explicite. Le code principal est écrit par Grégoire.
2. **Claude ne prend jamais de décision.** Son rôle se limite à deux actions :
   - **proposer** plusieurs options techniques (au moins deux), avec leurs avantages, leurs inconvénients et éventuellement une recommandation ;
   - **comparer** des options techniques proposées par Grégoire.

   Le choix final revient toujours à Grégoire. Il peut retenir l'une des options proposées par Claude, ou une autre qui n'en fait pas partie. Une recommandation de Claude n'est qu'un avis : elle n'engage pas le projet tant que Grégoire ne l'a pas choisie. Toutes les décisions, avec leur justification, sont dans `choix_techniques.md`.
3. **Les informations sont vérifiées et sourcées.** Les fiches distinguent les faits officiels, les sources tierces, les sources communautaires et les hypothèses, avec les étiquettes `[OFFICIEL]`, `[TIERS-PUBLIÉ]`, `[COMMUNAUTÉ]` et `[À VÉRIFIER]`.
4. **Aucun résultat n'est inventé.** Tous les chiffres de performance du projet viennent de scripts exécutés par Grégoire sur sa machine.
5. **Les corrections de code sont ciblées.** Claude analyse le code existant et indique les modifications précises plutôt que de le réécrire.

## 3. Usages par type

### 3.1 Pédagogie

Claude a été utilisé pour apprendre la diarisation depuis zéro :

- pipeline complète ;
- STFT, spectrogramme, MFCC ;
- VAD ;
- embeddings et clustering ;
- RTTM, UEM ;
- DER et JER, ainsi que leurs variantes (collar, overlap) ;
- interprétation des métriques.

Supports produits :

- un site de révision interactif, avec un schéma cliquable de la pipeline en 8 étapes ;
- une fiche théorique sur l'évaluation ;
- un exercice de calcul du DER à la main.

### 3.2 Recherche documentaire

- Fiches des corpus AMI et VoxConverse : accès, licence, langue, volume, conditions d'enregistrement, qualité des annotations, risques de fuite.
- État de l'art des corpus de diarisation accessibles.
- Vérification sur PyPI de la compatibilité des bibliothèques avec Python 3.14.
- Conditions d'accès aux modèles pyannote sur Hugging Face.
- Identification du risque de fuite lié à l'entraînement de `pyannote/segmentation-3.0`.

### 3.3 Aide à la décision

Pour chaque choix (corpus, split, méthode de tirage, collar, agrégation…), Claude a soit proposé des options, soit comparé celles que Grégoire avait en tête, en présentant leurs avantages et leurs inconvénients. La décision et sa justification reviennent à Grégoire. Certaines décisions ne correspondent d'ailleurs à aucune option proposée par Claude, ou en sont une variante, par exemple le tirage stratifié emboîté sur trois critères.

### 3.4 Rédaction de documentation

- Fiches de corpus et fichier de décision du corpus.
- Fiches de bibliothèques (pyannote, pyannote.metrics), dont les exemples ont été exécutés par Claude dans son propre environnement.
- Planning de la semaine 1.
- Ce fichier et `choix_techniques.md`, pré-remplis à partir des décisions prises ensemble.

### 3.5 Aide au code et à l'environnement

- Commandes d'installation et de configuration : WSL, venv, GitHub CLI, torch/CUDA, téléchargement avec `wget -c` et vérification MD5.
- Explication de bibliothèques (librosa, pyannote.core, pyannote.metrics).
- Débogage à partir des messages d'erreur, par exemple `externally-managed-environment`.
- Correction ou complétion de fonctions écrites par Grégoire (détails dans le tableau ci-dessous).

## 4. Provenance des fichiers

Légende :

- **G** : écrit par Grégoire ;
- **C** : rédigé par Claude, puis relu et validé par Grégoire ;
- **G + C** : écrit par Grégoire, avec des corrections ou des compléments de Claude.

### Code

| Fichier | Provenance | Détail |
| --- | --- | --- |
| `experimenations.py` | G | exploration audio, forme d'onde, spectrogramme, visualisation de la VAD |
| `plot_reference.py` | G | timeline de référence |
| `rttm_files.py` | G + C | `get_representative_files` : G. `get_representative_sample` : écrite par G, corrigée et complétée par C (tirage proportionnel par la méthode des plus grands restes, `random.Random(seed)`) |
| `Data_vizualizer.py` | *C** **(**non re**lu)* | provenance du bloc `plot_sample_vs_dev` à préciser |
| `metrics_and_sanity_checks.py` | G | sanity checks 1 et 2 |
| `energy_vad` | G | en cours |

### Documentation du dépôt

| Fichier | Provenance | Détail |
| --- | --- | --- |
| `README.md` | *G** **+** **C* |  |
| `docs/corpus/ami.md`, `docs/corpus/voxconverse.md` | C | grille de critères définie par G ; section « Observations personnelles » à compléter par G |
| `docs/decision_corpus.md` | C | décision et justification de G |
| `docs/notes-importantes-et-difficultes.md` | *G* |  |
| `docs/choix_techniques.md` | C | pré-rempli à partir des décisions de G, structure choisie par G |
| `docs/utilisation_claude.md` | C | structure choisie par G |

### Supports hors dépôt (artefacts Claude)

| Support | Provenance |
| --- | --- |
| Planning semaine 1 | C, validé par G |
| Site de révision Who Spoke | C |
| Fiche pyannote | C |
| Fiche pyannote.metrics | C |
| Fiche « Évaluer un système de diarisation (théorie) » | C |
| État de l'art des corpus de diarisation | C |
| Fiche « Terminal Ubuntu vs terminal VS Code » | C, complétée au fil des questions |

## 5. Limites et vérifications

- Claude peut se tromper ou s'appuyer sur des informations dépassées.
  - Les informations susceptibles d'évoluer (API, versions, licences, accès aux modèles) sont vérifiées sur les sources officielles, et datées dans les fiches.
  - Les points non confirmés sont marqués `[À VÉRIFIER]`.
- Les contrôles faits par Grégoire sur ses propres données comptent davantage que les affirmations de Claude. Exemples :
  - comptage des fichiers après téléchargement ;
  - sommes MD5 ;
  - écoute du fichier pqmho ;
  - sanity checks de la chaîne d'évaluation.
- Les choix de conception, l'exécution des expériences et l'interprétation finale des résultats relèvent de Grégoire.
