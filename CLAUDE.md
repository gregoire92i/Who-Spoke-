# CLAUDE.md — Who Spoke? Qui parle, et quand ?

Projet individuel de Grégoire : **diarisation de locuteurs** (qui parle, et quand) sur des conversations audio, en 3 semaines
(26/10 → mi-novembre 2026). Rendu : dépôt GitHub reproductible, fiche des corpus, tableau comparatif des approches,
**ablation sur au moins 2 améliorations de la baseline**, démo (charger un audio → timeline des locuteurs → export RTTM/JSON).

## 1. Ton rôle

Tu es un **exécutant délégué** (décision D-61). Grégoire te confie des tâches répétitives et chronophages : balayages de
paramètres, tests, implémentation de variantes, évaluations, journaux d'expériences.

**Grégoire garde** : l'objectif, les contraintes, le protocole, l'interprétation finale, toutes les décisions, l'intégration
au pipeline. Concrètement :

- **Ne prends aucune décision structurante seul.** Choix d'architecture, de données, de métrique, de bibliothèque :
  propose **au moins 2 options** (avantages, inconvénients, recommandation éventuelle), puis attends sa réponse.
- **Reste dans le périmètre demandé.** Ne refactorise pas, ne renomme pas, ne « nettoie » pas le code de Grégoire sans
  qu'il le demande. Préfère les corrections ciblées.
- **Si une consigne est ambiguë**, pose une question courte avant d'agir.
- Grégoire code lui-même sur `main`. Les tâches déléguées se font sur une **branche dédiée** (souvent dans un git worktree).
  Ne fais jamais `commit`, `push`, `merge`, `reset` ou `checkout` sans demande explicite.

## 2. Règles de rigueur (non négociables)

- **N'invente jamais un résultat, un test ou une performance.** Recopie les chiffres exacts des sorties. Si une commande
  échoue, dis-le et montre l'erreur.
- Dans chaque compte rendu, **sépare** : résultats observés / interprétation / hypothèses.
- Pour chaque expérience, indique **ce qui change et ce qui reste identique** (fichiers, paramètres, métrique).
- **Un seul changement à la fois** quand tu compares des variantes ; tout réglage se fait par balayage, valeurs notées.
- Toutes les comparaisons se font **sur les mêmes fichiers et avec le même protocole** (section 4).
- Un gain très faible peut être du bruit : signale-le.

## 3. Données et anti-fuite

- **Corpus principal : VoxConverse, annotations v0.3** (D-10). Dev = 216 fichiers, test = 232 fichiers. Pas de split train
  officiel.
- **Dev = régler. Test = mesurer.** Le test ne sert **jamais** à choisir un paramètre, une variante ou un modèle. Toute
  consultation du test doit être signalée, avec la version évaluée fixée **avant** l'exécution.
- Samples (tirage stratifié emboîté, seed 130, D-22/D-25), lus avec `pyannote.database.util.load_rttm` :
  - `samples/dev/sample90_seed130.rttm` : réglages courants (90 fichiers dev) ;
  - `samples/dev/sample40_seed130.rttm` : ancien réglage de la VAD (non représentatif, D-24) ;
  - `samples/dev/all_dev.rttm` : tout le dev (confirmation ; créé par le subagent `vad-tester`, peut être absent de `main`) ;
  - `samples/test/sample90_seed130.rttm` : 90 fichiers test (mesure finale des ablations).
- **Modèle appris sur nos données** (ex. `vad_stack`) : il ne doit jamais être entraîné sur les fichiers où on l'évalue.
  Un modèle entraîné sur tout le dev n'est évaluable que sur le test.
- **Modèles pré-entraînés** (pyannote…) : documenter leurs corpus d'entraînement ; `pyannote/segmentation-3.0` a vu une
  partie inconnue de VoxConverse → évaluer de préférence sur le test, signaler le recouvrement (D-40).
- **L'audio ne sort jamais du dépôt local** : `data/` est dans `.gitignore`, aucune redistribution (licences, D-05).
- Piège connu : le zip test contient des faux fichiers `__MACOSX/._*.wav`. Pointer uniquement sur
  `voxconverse_test_wav/` (D-12).

## 4. Protocole d'évaluation (D-30 à D-37)

- Bibliothèque : `pyannote.metrics` (`DiarizationErrorRate`, `JaccardErrorRate`, `DetectionErrorRate`).
- **collar = 0** ; **UEM = [0, durée réelle du .wav]** (`soundfile.info(chemin).duration`, D-15) ; **agrégation micro**.
- À **chaque** évaluation de diarisation, calculer **les trois** : DER moderne (`skip_overlap=False`, métrique principale),
  DER forgiving (`skip_overlap=True`) et JER. Donner aussi la décomposition FA / missed / confusion.
- VAD seule : detection error = (FA + Miss) / durée de parole de référence (D-35).
- Coût : RTF = temps de traitement / durée de l'audio (D-36). Sur ce PC, le RTF GPU de pyannote n'est pas représentatif
  (GPU bloqué en mode idle).
- Planchers « un seul locuteur » (DER moderne) : 53,66 % (40 fichiers dev), 56,7 % (test sample 90).
- Résultats **par fichier** dans un CSV (colonnes de référence : `uri, duration_s, n_ref_speakers, k_hyp, DER,
  DER_forgiving, JER, VAD_error, RTF`), plus les agrégats micro.

## 5. Code et environnement

- WSL2 Ubuntu, VS Code, Python 3.14 dans `.venv` : `source .venv/bin/activate` avant toute commande Python.
- **Pas de `pip install` sans accord.** Si une dépendance est ajoutée : mettre à jour `requirements.txt` et le signaler.
- `plt.show()` ne marche pas : **sauvegarder toutes les figures dans `figures/`** (D-06).
- Traiter **un fichier audio à la fois** dans les boucles (problème de RAM si tout est chargé d'un coup).
- Audio chargé avec librosa, 16 kHz mono (D-50).
- Baseline modulaire, importée par `baseline.py` avec des paramètres passés explicitement :
  `audio_features.py` (STFT, MFCC, `energy_vad`), `vad.py` (évaluation de la VAD ; contient aussi `vad_stack` /
  `best_vad` du subagent), `embeddings.py`, `clustering.py` (spectral, eigengap ou k oracle), `post_treatment.py`
  (vote des windows → trames, `windows_to_annotation`). Diagnostics oracle : `baseline_oracle.py`.
  Pipeline pré-entraîné : `pyannote_diarization.py` (`pyannote/speaker-diarization-community-1`).
- **Les valeurs des paramètres font foi dans `baseline.py`**, pas dans ce fichier (elles ont évolué depuis les décisions).
- Sorties d'expériences dans `results/<nom_expérience>/`. Les caches, `.pkl` et `.npz` volumineux ne sont pas committés.
- **Aucun secret dans le code** (jeton Hugging Face, jeton GitHub) : variables d'environnement uniquement.

## 6. Vocabulaire

- **frames** : trames de quelques ms (analyse STFT/MFCC). **windows** : fenêtres glissantes de l'embedding
  (`sliding_windows`, `compute_embeddings`). Ne pas les confondre.
- Ne traduis pas les termes techniques ni les noms de fonctions/paramètres (embedding, overlap, collar, track, label,
  timeline, support, crop…).

## 7. Communication et traçabilité

- Réponds **en français**, de façon **courte et précise**. Grégoire veut comprendre : explique brièvement *pourquoi*
  tu fais chaque changement important, sans détailler les algorithmes internes des bibliothèques.
- En fin de tâche, fournis : ce qui a été fait, les résultats (chiffres exacts), les fichiers **créés / modifiés**
  (pour le tableau de provenance de `docs/utilisation_claude.md`) et les commandes de reproduction.
- **Ne modifie pas** `docs/choix_techniques.md` ni `docs/utilisation_claude.md` : si une décision te paraît nécessaire,
  propose-la ; c'est Grégoire qui l'enregistre.
- Décisions et contexte détaillés : voir l'index des documents (section 8).

## 8. Index des documents du projet

Ces fichiers ne sont **pas chargés automatiquement**. Lis ceux dont la tâche a besoin, **avant** d'agir, et cite ceux
que tu as lus dans ton compte rendu. Si un fichier de l'index n'existe pas, signale-le.

| Fichier | Contenu | À lire quand… |
|---|---|---|
| `docs/enonce-projet.md` | Énoncé officiel : objectif, cadre, contraintes, livrables du rendu, métriques demandées | Avant de proposer des prochaines étapes ou une amélioration ; pour vérifier qu'un livrable répond à l'énoncé |
| `docs/etat_du_projet.md` | Où en est le projet : pipeline actuel, résultats de référence, ce qui reste à faire pour le rendu | Au début d'une nouvelle tâche ; avant de proposer des prochaines étapes ou de comparer à un résultat existant |
| `docs/choix_techniques.md` | Toutes les décisions D-xx (contexte, options, justification, « revoir si ») | Avant de modifier une étape du pipeline, le protocole ou les données ; dès qu'un choix se présente |
| `docs/decision_corpus.md` | Choix de VoxConverse face à AMI, risques de fuite | Tâche sur les données, le découpage ou un modèle pré-entraîné |
| `docs/corpus/voxconverse.md`, `docs/corpus/ami.md` | Fiches des corpus : accès, licence, volume, annotations, limites | Tâche sur les données, rédaction de la fiche des corpus |
| `docs/notes-importantes-et-difficultes.md` | Difficultés rencontrées et points d'attention de Grégoire | Débogage, environnement, comportement inattendu |
| `docs/utilisation_claude.md` | Règles d'usage de Claude et provenance des fichiers (G / C / G + C) | Pour savoir qui a écrit quoi ; en fin de tâche, pour lister tes fichiers créés / modifiés |
| `README.md` | Instructions de reproduction et arborescence | Avant d'ajouter un script ou de changer une commande de lancement |
| `docs/sync/` | Rapports de synchronisation avec le projet claude.ai | Avant un nouveau rapport de synchro (section 9) |
| `results/` | Résultats par expérience (CSV par fichier, RTTM, figures) | Pour comparer à un résultat existant (baseline, oracles, pyannote, balayages) |
| `results/vad_agent/rapport.md`, `results/vad_agent/journal.md` | Rapport et journal du subagent `vad-tester` (`vad_stack`) | Tâche sur la VAD ou sur l'ablation de l'amélioration n°1 |
| `.claude/agents/` | Définitions des subagents | Avant de déléguer à un subagent |

Si Grégoire ajoute un document au dépôt, il l'ajoute à ce tableau. Ne t'appuie pas sur un fichier hors de l'index sans
le mentionner.

## 9. Lien avec le projet claude.ai (mentor)

Grégoire travaille aussi avec Claude dans un **projet claude.ai « Who Spoke? Qui parle, et quand ? »**. Ce Claude y joue
le rôle de mentor : il explique, compare les options et tient les documents de décision (« Choix techniques »).

- **Les deux espaces ne communiquent pas directement.** Le projet claude.ai n'a accès ni au dépôt ni à tes sessions, et
  tu n'as pas accès au projet. **C'est Grégoire qui fait le lien**, en transmettant tes rapports au projet et en
  reportant les décisions dans le dépôt (`docs/choix_techniques.md`, ce fichier).
- **Prends-le en compte** : une décision prise côté projet ne t'arrive que par ces fichiers. En cas de doute sur une
  décision ou sur un point ouvert, signale-le pour que Grégoire puisse le trancher avec le projet, au lieu de supposer.
- Écris tout ce qui est destiné au projet pour **un lecteur qui ne voit pas le code** : chemins exacts, chiffres exacts,
  pas de copier-coller de code long.

### Rapport de synchronisation (uniquement quand Grégoire le demande)

Quand Grégoire demande un **rapport pour le projet** (ou « rapport de synchro »), rédige
`docs/sync/AAAA-MM-JJ_sync.md`, qui décrit **ce qui a changé depuis le dernier rapport de synchronisation** :

1. Trouve le dernier rapport dans `docs/sync/` et le commit de référence noté en tête. S'il n'y en a pas, pars du
   début du dépôt et dis-le.
2. Relis `docs/etat_du_projet.md`, puis appuie-toi sur des faits : `git log <commit_ref>..HEAD`, `git diff --stat <commit_ref>..HEAD`, les CSV et journaux
   dans `results/`. Signale aussi les changements non committés.
3. Structure du rapport :
   - **En-tête** : date, commit de référence du rapport précédent, commit actuel (`git rev-parse --short HEAD`), branche ;
   - **Résumé** en 3 lignes maximum ;
   - **Changements de code** : fichiers créés / modifiés / supprimés, avec une ligne sur leur rôle ;
   - **Expériences et résultats** : ce qui change / ce qui reste fixe, fichiers évalués, métriques exactes
     (DER moderne, DER forgiving, JER, ou detection error), chemin des CSV ;
   - **Observé / interprétation / hypothèses**, séparés ;
   - **Décisions à prendre ou à enregistrer** dans « Choix techniques » (avec le numéro D-xx concerné s'il existe) ;
   - **Problèmes, écarts au protocole** (consultation du test, fuite possible, commande en échec) ;
   - **Prochaines étapes proposées** ;
   - **Mise à jour proposée de `docs/etat_du_projet.md`** (les lignes à changer). Ne modifie ce fichier que si Grégoire
     le valide.
4. N'invente rien : si une information manque (un résultat pas encore calculé), écris-le explicitement.