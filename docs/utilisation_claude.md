# Utilisation de Claude dans le projet Who Spoke ?

> Ce document explique comment l'assistant IA Claude (Anthropic) a été utilisé dans ce projet, et ce qui vient de Grégoire ou de Claude. Dernière mise à jour : 14/11/2026 (ajout de Claude Code, sections 1, 3.6 et 4 ; mise à jour faite par Claude Code à la demande de Grégoire).

## 1. Outil

- **Assistant** : Claude, développé par Anthropic, utilisé via claude.ai dans un « Projet » dédié qui partage les fichiers de contexte entre les conversations.
- **Modèle(s)** : *O**p**u**s** **5.5** **et **S**o**nnet** **5**.**5** *(le modèle peut changer d'une conversation à l'autre).
- **Données transmises à Claude** :
  - des extraits de code ;
  - des sorties de terminal ;
  - des fichiers de documentation du projet.
  - **Aucun fichier audio du corpus** ne lui a été transmis.
- **Claude Code** (depuis le 12/11/2026, décision D-61) : l'agent en ligne de commande d'Anthropic, exécuté **localement** dans le dépôt (WSL2), avec le modèle Opus 5.5. Il lit et exécute le code sur la machine de Grégoire : l'audio reste sur la machine et n'est jamais envoyé vers un service externe (D-05). Ses règles sont dans `CLAUDE.md` ; les subagents spécialisés sont définis dans `.claude/agents/` (`vad-researcher`, `clustering-embeddings-analyzer`, `demo-builder`).
- **Les deux espaces ne communiquent pas** : le projet claude.ai (mentor) ne voit ni le dépôt ni les sessions Claude Code ; Grégoire fait le lien (rapports de synchronisation dans `docs/sync/`).

## 2. Règles d'usage

Ces règles ont été fixées au début du projet, dans les instructions de celui-ci :

1. **Claude est un mentor, pas un générateur de solutions.** Il explique, compare et corrige. Il ne génère jamais de code sans instruction explicite. Le code principal est écrit par Grégoire.
2. **Claude ne prend jamais de décision.** Son rôle se limite à deux actions :
   - **proposer** plusieurs options techniques (au moins deux), avec leurs avantages, leurs inconvénients et éventuellement une recommandation ;
   - **comparer** des options techniques proposées par Grégoire.

   Le choix final revient toujours à Grégoire. Il peut retenir l'une des options proposées par Claude, ou une autre qui n'en fait pas partie. Une recommandation de Claude n'est qu'un avis : elle n'engage pas le projet tant que Grégoire ne l'a pas choisie. Toutes les décisions, avec leur justification, sont dans `choix_techniques.md`.
3. **Les informations sont vérifiées et sourcées.** Les fiches distinguent les faits officiels, les sources tierces, les sources communautaires et les hypothèses, avec les étiquettes `[OFFICIEL]`, `[TIERS-PUBLIÉ]`, `[COMMUNAUTÉ]` et `[À VÉRIFIER]`.
4. **Aucun résultat n'est inventé.** Tous les chiffres de performance du projet viennent de scripts exécutés par Grégoire sur sa machine.
   - *Mise à jour de la règle (14/11/2026) :* depuis l'arrivée de Claude Code (12/11/2026, D-61), les chiffres viennent de scripts exécutés sur la machine de Grégoire, **par lui ou par Claude Code**. Dans ce second cas, les chiffres sont recopiés tels quels des sorties, et les commandes de reproduction sont données dans les rapports.
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

### 3.6 Exécution déléguée (Claude Code)

Rôle d'**exécutant délégué** (D-61) : Grégoire fixe l'objectif, le protocole et les contraintes, prend toutes les décisions et interprète les résultats ; Claude Code exécute les tâches répétitives et rend compte (résultats observés / interprétation / hypothèses séparés, chiffres recopiés des sorties).

- **VAD** (subagent `vad-researcher`) : amélioration itérative de la VAD (`vad_stack` / `best_vad` dans `vad.py`), journal et rapport dans `results/vad_agent/`.
- **Analyse du clustering** (subagent `clustering-embeddings-analyzer`) : diagnostic « k oracle pire que l'eigengap », scripts dans `analysis/clustering_agent/`, résultats dans `results/diag_k_oracle_vs_eigengap/`.
- **Démonstration** (subagent `demo-builder`, 14/11) : application Gradio dans `demo/`, construite par phases validées par Grégoire (cartographie, propositions, cœur, interface, tests, répétition) ; choix de Grégoire : intégration de la baseline par un argument `vad_train` (B3), Gradio + plotly, export RTTM uniquement, pyannote hors ligne. Rapport, journal et architecture dans `results/demo_agent/`.
- **Petites tâches** : correction ciblée de `baseline.py` (argument `vad_train`), mesures du RTF de la baseline et analyse de la complexité du clustering NME (notées dans `docs/notes-importantes-et-difficultes.md`).

Contrairement à l'usage via claude.ai, Claude Code **exécute** des scripts et produit des chiffres sur la machine de Grégoire ; ces chiffres sont recopiés des sorties et les commandes de reproduction sont données dans les rapports.

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
| `baseline.py` | G + C | argument `vad_train` de `diarize_file` ajouté par Claude Code (14/11), à la demande de G |
| `vad.py` | G + C | `vad_stack` / `best_vad` : subagent `vad-researcher` (C) ; reste : G |
| `analysis/clustering_agent/*.py` | C | subagent `clustering-embeddings-analyzer` |
| `demo/__init__.py`, `demo/core.py`, `demo/baseline_adapter.py`, `demo/pyannote_adapter.py`, `demo/plots.py`, `demo/app.py`, `demo/style.css` | C | subagent `demo-builder`, choix de conception de G (phase 1) |
| `tests/test_demo.py` | C | subagent `demo-builder` (16 tests) |
| `.gitignore`, `requirements.txt` | G + C | lignes de la démo (`demo/tmp/`, `.gradio/`, extensions audio) et dépendances pytest / gradio / plotly ajoutées par C avec l'accord de G |
| `clustering.py` (`nme_spectral_clustering`), `nme_sc_experiment.py`, `ablation.py` | *à préciser par G* | |

### Documentation du dépôt

| Fichier | Provenance | Détail |
| --- | --- | --- |
| `README.md` | *G** **+** **C* |  |
| `docs/corpus/ami.md`, `docs/corpus/voxconverse.md` | C | grille de critères définie par G ; section « Observations personnelles » à compléter par G |
| `docs/decision_corpus.md` | C | décision et justification de G |
| `docs/notes-importantes-et-difficultes.md` | G + C | sections marquées « noté par Claude Code » : C |
| `CLAUDE.md`, `.claude/agents/*.md` | *à préciser par G* | |
| `docs/rapport-ablation.md` | *à préciser par G* | |
| `results/vad_agent/rapport.md`, `results/vad_agent/journal.md` | C | subagent `vad-researcher` |
| `results/diag_k_oracle_vs_eigengap/rapport.md`, `journal.md`, `guide_figures.md` | C | subagent `clustering-embeddings-analyzer` |
| `results/demo_agent/rapport.md`, `journal.md`, `architecture.md` | C | subagent `demo-builder` |
| `figures/demo/*.png` | C | subagent `demo-builder` (fichiers dev uniquement) |
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
