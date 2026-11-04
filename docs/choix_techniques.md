# Choix techniques — Who Spoke ?

> Journal des décisions techniques du projet, organisé par thème. Dernière mise à jour : 04/11/2026.

## Mode d'emploi

Chaque décision a un identifiant (`D-xx`), qui ne change jamais, même si la décision est remplacée. On l'utilise pour faire référence à une décision depuis le code, le README ou le rapport.

Format d'une entrée :

- **Date** : date de la décision, et des révisions éventuelles.
- **Statut** : l'une des valeurs suivantes.
  - `actif` : la décision s'applique.
  - `provisoire` : la décision a été prise « pour l'instant » et sera revue.
  - `remplacé par D-xx` : la décision ne s'applique plus.
- **Contexte** : le problème qu'on devait résoudre.
- **Options** : les alternatives considérées.
- **Décision** : ce qu'on a choisi.
- **Justification** : pourquoi on l'a choisi.
- **Revoir si** : ce qui nous ferait changer d'avis.

Les champs marqués *à compléter* n'ont pas été notés au moment de la décision.

⚠️ Deux décisions sont **recopiées intégralement** depuis d'autres fichiers :

- D-10, recopiée depuis `docs/decision_corpus.md` ;
- D-30 à D-37, recopiées depuis le fichier « Décisions d'évaluation ».

À chaque modification, il faut mettre à jour **les deux fichiers**.

## 1. Environnement de développement

### D-01 — Environnement de développement : VS Code + WSL2

- **Date** : 26/10/2026
- **Statut** : actif
- **Contexte** : il fallait choisir où écrire et exécuter le code pendant trois semaines.
- **Options** :
  - VS Code + WSL2 (Ubuntu) en local ;
  - Google Colab.
- **Décision** : VS Code avec l'extension WSL, sur Ubuntu sous WSL2. Le dépôt est dans le système de fichiers Linux (`~/projets/Who-Spoke-`). Colab reste possible plus tard pour du GPU.
- **Justification** : *à compléter*
- **Revoir si** : les besoins en GPU dépassent ce que permet la machine locale.

### D-02 — Gestion du dépôt : GitHub, authentification par GitHub CLI

- **Date** : 26/10/2026
- **Statut** : actif
- **Contexte** : il fallait authentifier Git sur GitHub depuis WSL.
- **Options** : *à compléter*
- **Décision** :
  - dépôt `gregoire92i/Who-Spoke-`, branche `main` ;
  - authentification via GitHub CLI (`gh`), en HTTPS ;
  - le jeton n'est jamais écrit en clair.
  - Mise à jour du 03/11/2026 : le dépôt est passé en public, et le mentor a été ajouté comme collaborateur.
- **Justification** : *à compléter*

### D-03 — Python 3.14 dans un environnement virtuel `.venv`

- **Date** : 26–27/10/2026
- **Statut** : provisoire
- **Contexte** : il fallait isoler les dépendances du projet et rendre l'installation reproductible.
- **Options** :
  - Python 3.14 système ;
  - repli sur Python 3.12 via uv ou conda si une bibliothèque n'est pas compatible.
- **Décision** :
  - `.venv` créé dans le dépôt, avec Python 3.14.4 ;
  - dépendances figées avec `pip freeze > requirements.txt`, puis committées.
- **Justification** : c'est la version déjà installée. Au 26/10, librosa, numba et torch déclaraient supporter 3.14 sur PyPI.
- **Revoir si** : NeMo, SpeechBrain ou un modèle pyannote ne fonctionnent pas sous 3.14. Ce test de bout en bout n'a pas encore été fait.

### D-04 — Installation de torch et de la pile pyannote : installation standard

- **Date** : 01/11/2026
- **Statut** : actif
- **Contexte** : il fallait vérifier que la pile deep learning s'installe sous Python 3.14, avec le GPU (RTX 3050 Laptop, CUDA visible dans WSL).
- **Options** :
  - A : installation standard depuis PyPI ;
  - autres options : *à compléter*.
- **Décision** : option A. Versions installées dans le `.venv` :
  - torch 2.15.0+cu132 ;
  - pyannote.audio 4.0.7 ;
  - pyannote.metrics 4.2.0 ;
  - pyannote.core / pyannote.database ;
  - torchcodec 0.17.0+cu130.
- **Justification** : `cuda.is_available()` renvoie `True`, et un calcul GPU a été vérifié. `requirements.txt` a été mis à jour et committé.
- **Revoir si** : le test de bout en bout avec un vrai modèle pyannote (accès Hugging Face) échoue.

### D-05 — Données exclues du dépôt Git

- **Date** : 27/10/2026
- **Statut** : actif
- **Contexte** : les licences ne permettent pas de redistribuer l'audio : droits des vidéos pour VoxConverse, droits de la personnalité pour AMI. Les fichiers sont en plus volumineux.
- **Décision** : le dossier `data/` est dans `.gitignore`. Les figures sont sauvegardées dans `figures/`.
- **Justification** : respect des licences et des droits des personnes, et taille du dépôt.

### D-06 — Affichage des figures : sauvegarde dans des fichiers

- **Date** : 02/11/2026
- **Statut** : actif
- **Contexte** : `plt.show()` ne fonctionne pas dans l'environnement WSL.
- **Décision** : toutes les figures sont sauvegardées dans `figures/` au lieu d'être affichées.

## 2. Données et corpus

### D-10 — Corpus principal : VoxConverse (annotations v0.3)

- **Date** : 28/10/2026
- **Statut** : actif. Décision révisable si les observations (téléchargement, écoute, timelines) révèlent un problème bloquant.
- **Contexte** : il fallait choisir le corpus qui servira à construire le découpage dev / validation, à régler la baseline et à comparer les approches.
- **Options** :
  - AMI Meeting Corpus : environ 100 h de réunions en anglais, enregistrées dans 3 salles équipées ;
  - VoxConverse : environ 64 h d'extraits de vidéos YouTube (débats, journaux, talk-shows).
- **Décision** : VoxConverse, annotations v0.3.

**Tableau comparatif**

| Critère | AMI | VoxConverse |
| --- | --- | --- |
| **Accès** | Libre, choix réunion par réunion et flux par flux. RTTM non fournis par les auteurs : il faut passer par le dépôt tiers AMI-diarization-setup | Libre. Deux archives zip (dev, test) avec somme MD5. **RTTM fournis directement** par les auteurs (v0.3) |
| **Licence** | CC BY 4.0 (page officielle actuelle) ; licence non commerciale dans des sources plus anciennes. Participants avec formulaire de consentement | CC BY 4.0 « for research use » ; **droit d'auteur des vidéos conservé par leurs propriétaires**. Pas de consentement individuel (personnalités publiques) |
| **Langue** | Anglais, majorité de locuteurs non natifs | Majoritairement anglais selon des sources secondaires ; **non précisé officiellement** |
| **Volume** | Environ 100 h ; splits train / dev / test disponibles (136 / 18 / 16 réunions) | 448 fichiers, environ 64 h (dev 216, test 232) ; **pas de split train** |
| **Locuteurs par enregistrement** | 3 à 5, presque toujours 4 | 1 à 21 (moyenne 4,5 en dev, 6,5 en test) |
| **Conditions d'enregistrement** | Contrôlées : 3 salles, participants assis, micros-casques et micros de table. Deux tiers de jeux de rôle scénarisés | Non contrôlées : bruit, musique, rires, montage TV, distances au micro variables. **Non scénarisé** |
| **Chevauchement de parole** | Élevé : environ 16 % (étude tierce, même méthode que pour VoxConverse) | Faible en comparaison : 3,1 à 3,8 % du temps de parole en moyenne (auteurs), 3,5 % (même étude tierce), de 0 à environ 30 % selon les fichiers |
| **Qualité des annotations** | Transcription manuelle, mais instants des mots obtenus par alignement automatique. Frontières potentiellement « larges » (pauses incluses dans les segments, cf. arXiv 2507.09226) | Semi-automatique puis vérification humaine ; frontières à 0,1 s près, coupure aux pauses de plus de 0,25 s ; désaccord entre deux annotateurs d'environ 1 % de DER |
| **Répétition des locuteurs** | **Structurelle au sein d'une série** : une même équipe de 4 mène les 4 réunions a, b, c, d d'une série scénarisée (fortement probable, non écrit explicitement, à vérifier dans les RTTM). **Entre séries** : groupes différents (≈ 30 répétitions du scénario), un recouvrement avéré (IB4005), le reste non documenté | Possible pour des personnalités publiques, **non documentée** \[À VÉRIFIER\] |
| **Recouvrement avec l'entraînement de pyannote** | Oui (corpus listé), splits non précisés | Oui (corpus listé), **fichiers utilisés inconnus** |

**VoxConverse par rapport à AMI**

- Avantages :
  - **Variété** : sources, émissions, ambiances sonores et nombre de locuteurs (1 à 21) bien plus diversifiés. C'est plus proche d'enregistrements inconnus comme ceux de la démo finale.
  - **Conditions réelles, non scénarisées**, contre des jeux de rôle en salle contrôlée.
  - **Pas de répétition structurelle des locuteurs** entre enregistrements, contrairement aux séries d'AMI. Le risque de fuite lors du découpage dev / validation est a priori plus faible, mais pas nul (personnalités publiques récurrentes, non vérifié).
  - **RTTM fournis directement par les auteurs** : aucune conversion ni dépôt tiers.
  - **Frontières plus précises**, avec des règles d'annotation pensées pour la diarisation.
  - Taille raisonnable des fichiers du dev (5,6 min en moyenne), ce qui est pratique pour itérer vite.
- Inconvénients :
  - **Droits sur l'audio moins clairs** : le droit d'auteur des vidéos reste à leurs propriétaires. Pas de redistribution, à mentionner dans le rapport.
  - **Langue non documentée officiellement.**
  - **Pas de split train** : gênant seulement si on fait du fine-tuning en semaine 3.
  - **Annotations semi-automatiques**, avec des erreurs connues sur les locuteurs hors champ ; une correction v0.3 a déjà été nécessaire.
  - **Chevauchement faible en moyenne** : teste moins les réunions animées où plusieurs personnes parlent en même temps.
  - **Fichiers d'entraînement de pyannote inconnus** : plus difficile de garantir une comparaison équitable qu'avec AMI.
  - Archives à télécharger en entier (plusieurs Go) même si on n'utilise que 10 à 15 fichiers.

**AMI par rapport à VoxConverse**

- Avantages :
  - **Consentement explicite** des participants et licence plus claire sur l'audio.
  - **Plusieurs versions audio de la même réunion** (micros-casques, micro de table) : on peut mesurer l'effet de la qualité du micro, utile pour une ablation.
  - **Splits train / dev / test officiels**, utiles en cas de fine-tuning.
  - **Chevauchement élevé**, représentatif des réunions réelles.
  - Téléchargement fichier par fichier, donc plus léger pour un petit sous-ensemble.
  - Corpus de référence très utilisé, avec de nombreux résultats publiés.
- Inconvénients :
  - **Mêmes locuteurs dans plusieurs enregistrements au sein d'une même série** : chaque équipe de 4 enregistre 4 réunions, ce qui crée un risque de fuite si on découpe par réunion et non par série. Entre séries, les groupes sont différents (un recouvrement avéré : IB4005).
  - **Peu de variété** : 3 salles, réunions de bureau, deux tiers de jeux de rôle, presque toujours 4 personnes. Risque de mal généraliser à la démo.
  - **Pas de RTTM officiels** : dépendance à un dépôt tiers.
  - **Frontières d'annotation potentiellement larges**, ce qui peut gonfler artificiellement la parole manquée dans le DER.
  - Chevauchement élevé : très difficile pour une baseline classique qui n'attribue qu'un locuteur à la fois.

**Justification**

1. **Variété des enregistrements.** VoxConverse est fait de vidéos prises sur Internet, non scénarisées, en conditions non maîtrisées, avec des types d'enregistrements plus variés. Le système sera jugé sur des enregistrements inconnus préparés par l'organisation : un corpus varié nous donne une mesure plus réaliste de sa capacité à généraliser.
2. **Risque de fuite de données dans AMI.** Dans AMI, la structure même du corpus fait que les mêmes personnes participent à plusieurs enregistrements : une équipe de 4 enregistre les 4 réunions d'une série scénarisée. C'est fortement probable d'après la description officielle, et à confirmer dans les RTTM. Un découpage dev / validation mal fait mettrait les mêmes voix des deux côtés, et nos résultats de validation seraient trop optimistes. Le corpus n'a en revanche pas été enregistré par un seul groupe : environ 30 répétitions du scénario avec des groupes différents.

**Points de vigilance**

- **La fuite liée aux locuteurs existe aussi, dans une moindre mesure, dans VoxConverse.** Des personnalités publiques peuvent apparaître dans plusieurs vidéos. À vérifier en écoutant les fichiers et en lisant les RTTM.
- **Côté AMI, ce risque était maîtrisable** en découpant par série plutôt que par réunion. Le choix de VoxConverse repose donc surtout sur la variété ; l'argument de fuite est un argument complémentaire.
- **Fuite avec les modèles pré-entraînés.** pyannote/segmentation-3.0 a été entraîné sur une partie inconnue de VoxConverse. Ce risque est plus fort qu'avec AMI, dont les splits sont mieux identifiés. Voir D-40.
- **Effet du découpage sur la baseline.** Pour une baseline non supervisée (MFCC et clustering), la présence de mêmes locuteurs dans dev et validation pèse surtout via le réglage des seuils. Elle deviendrait critique si on entraîne ou adapte un modèle.

**Revoir si**

- des annotations v0.3 manifestement fausses apparaissent sur une partie importante des fichiers écoutés ;
- une proportion importante de fichiers ne sont pas en anglais, si cela pose problème pour la démo ;
- un problème d'accès ou de téléchargement est bloquant.

Dans ces cas, AMI reste l'alternative documentée. Utiliser AMI en corpus secondaire est une option **non décidée**.

### D-11 — Source de téléchargement de VoxConverse

- **Date** : 28–29/10/2026
- **Statut** : actif
- **Contexte** : l'audio n'est pas disponible sur le site d'Oxford.
- **Décision** :
  - **Audio** : miroir KAIST.
    - `voxconverse_dev_wav.zip`, MD5 `2a6e07e7473d9841abb132554a698a36` ;
    - `voxconverse_test_wav.zip`, MD5 `834558bbd9b1ffd2d4893181556ceddd`.
  - **Annotations** : clone de `joonson/voxconverse`, branche `master` (v0.3), dans `data/voxconverse/annotations/`.
  - **Contrôle d'intégrité** : 216 fichiers `.wav` en dev et 232 en test, ce qui correspond aux chiffres officiels.
- **Justification** : c'est la seule source accessible, et les sommes MD5 sont fournies.

### D-12 — Dossier `__MACOSX` du zip test : conservé

- **Date** : 03/11/2026
- **Statut** : actif
- **Contexte** : le zip test contient 232 faux fichiers `._*.wav` (métadonnées macOS) dans `__MACOSX/`.
- **Options** :
  - supprimer le dossier ;
  - pointer les scripts uniquement sur `voxconverse_test_wav/`.
- **Décision** : le dossier est conservé. Les scripts doivent pointer sur le bon dossier.
- **Revoir si** : un script compte ou lit les faux fichiers. C'est un risque connu, à garder en tête.

### D-13 — Lecture des RTTM : `pyannote.database.util.load_rttm`

- **Date** : 01/11/2026
- **Statut** : provisoire
- **Contexte** : les lignes des RTTM de VoxConverse ne sont pas triées par temps.
- **Options** :
  - A : parser maison ;
  - B : `load_rttm` de pyannote.
- **Décision** : B.
- **Justification** :
  - `load_rttm` renvoie directement une `Annotation`, triée chronologiquement ;
  - c'est l'objet attendu par `pyannote.metrics`.

### D-14 — Visualisation de la référence : `pyannote.core.notebook.plot_annotation`

- **Date** : 01/11/2026
- **Statut** : provisoire
- **Options** :
  - A : matplotlib écrit à la main ;
  - B : `plot_annotation`.
- **Décision** : B. Script `plot_reference.py`.

### D-15 — Durée d'un fichier : durée réelle du `.wav`

- **Date** : 02/11/2026
- **Statut** : actif
- **Contexte** : le RTTM ne couvre pas forcément tout l'audio (silence ou musique au début ou à la fin).
- **Options** :
  - `extent()` de l'annotation RTTM ;
  - durée du fichier audio.
- **Décision** : `soundfile.info(chemin_wav).duration`.
- **Justification** : la durée sert à l'UEM (D-33), qui doit couvrir tout l'audio. Sinon, des fausses alarmes hors annotation ne seraient pas comptées.

## 3. Découpage des données

### D-20 — Pas de split train

- **Date** : 01/11/2026
- **Statut** : actif
- **Contexte** : VoxConverse n'a pas de split train officiel, et la baseline n'apprend aucun modèle.
- **Décision** : pas de split train pour l'instant.
- **Revoir si** : on décide de faire du fine-tuning (voir les décisions ouvertes).

### D-21 — Réglage sur un sous-ensemble fixe du dev, confirmation sur tout le dev, test à la fin

- **Date** : 01/11/2026, complétée le 03/11/2026
- **Statut** : actif
- **Contexte** : itérer sur 216 fichiers est trop long, mais régler sur trop peu de fichiers risque de ne pas être représentatif.
- **Options** :
  - A : tout le dev pour régler ;
  - B : un sous-ensemble fixe du dev pour itérer, puis tout le dev pour confirmer.
- **Décision** :
  - option B ;
  - le test VoxConverse est utilisé **en entier**, plus tard, pour l'évaluation finale.
- **Justification** : on itère vite, avec des réglages confirmés sur un ensemble plus large. Le test reste intact pour la fin.

### D-22 — Méthode de tirage du sous-ensemble : stratifié emboîté

- **Date** : 03/11/2026
- **Statut** : actif
- **Contexte** : Grégoire exige un sous-ensemble représentatif du dev.
- **Options** :
  - tirage aléatoire simple ;
  - tirage stratifié.
- **Décision** : tirage stratifié emboîté.
  - Les strates : 3 tiers de nombre de locuteurs × 3 tiers de taux d'overlap × 3 tiers de % de parole, soit 27 sous-batchs.
  - Le tirage est proportionnel, avec la méthode des plus grands restes.
  - Il utilise `random.Random(seed)`, avec `seed=42`.
  - La liste des fichiers est triée (`sorted`) avant le tirage.
  - Fonction : `get_representative_sample` dans `rttm_files.py`.
- **Justification** : la comparaison visuelle sample vs dev complet (`Data_vizualizer.py`, figures `sample_vs_dev_*.png`) montre des distributions très proches.

### D-23 — Pas de liste versionnée du sous-ensemble

- **Date** : 03/11/2026
- **Statut** : actif
- **Options** :
  - écrire la liste des fichiers tirés dans un fichier versionné ;
  - ne pas l'écrire.
- **Décision** : pas de liste versionnée.
- **Justification** :
  - la reproductibilité est assurée par la seed et le tri préalable ;
  - la taille du sample reste modifiable : 30 fichiers au départ, 40 pour les sanity checks.
- **Revoir si** : les fichiers du dossier changent, ce qui changerait le tirage à seed égale.

## 4. Protocole d'évaluation

> Recopié depuis le fichier « Décisions d'évaluation » rédigé par Grégoire. Ces décisions ont été prises le 03/11/2026, « pour l'instant ».

### D-30 — Bibliothèque de métriques : pyannote.metrics

- **Date** : 03/11/2026
- **Statut** : provisoire. Utilisée de fait, mais pas formellement tranchée.
- **Options** :
  - pyannote.metrics ;
  - dscore ou équivalent ;
  - implémentation maison.
- **Décision** : `pyannote.metrics` 4.2.0, avec `DiarizationErrorRate`.

### D-31 — Collar = 0

- **Date** : 03/11/2026
- **Statut** : provisoire
- **Historique** : 50 ms (convention NIST), puis 80 ms, puis 0.
- **Décision** : collar = 0.
- **Justification** : c'est le protocole des résultats publiés par pyannote sur VoxConverse et AMI, donc la comparaison avec ces résultats est directe.

### D-32 — DER moderne en métrique principale, DER forgiving toujours calculé en plus

- **Date** : 03/11/2026
- **Statut** : actif
- **Décision** :
  - métrique principale : DER avec overlap compté (`skip_overlap=False`) ;
  - **règle** : à **chaque** évaluation (sanity checks, baseline, toutes les expériences), on calcule les **deux** DER : moderne (`skip_overlap=False`) et forgiving (`skip_overlap=True`) ;
  - JER en métrique complémentaire.
- **Justification** :
  - le DER moderne est l'approche actuelle ;
  - le DER forgiving permet d'isoler l'effet de l'overlap, qu'une baseline mono-locuteur ne peut pas traiter.

### D-33 — UEM = \[0, durée réelle de l'audio\]

- **Date** : 03/11/2026
- **Statut** : actif
- **Décision** : tout le fichier est évalué, via la taille réel du fichier wav. La durée vient de D-15.

### D-34 — Agrégation multi-fichiers : micro

- **Date** : 03/11/2026
- **Statut** : actif
- **Options** :
  - micro : les erreurs de tous les fichiers sont cumulées puis divisées par la parole totale ;
  - macro : moyenne des DER par fichier.
- **Décision** : micro.
- **Justification** : jugé plus représentatif. Un fichier court avec très peu de parole ne fait pas exploser la moyenne, comme pqmho (DER de 831 %, voir D-37).

### D-35 — Évaluation de la détection de parole seule

- **Date** : 03/11/2026
- **Statut** : actif
- **Décision** : detection error = (FA + MISS) / durée de parole de référence.

### D-36 — Coût de calcul : RTF

- **Date** : 03/11/2026
- **Statut** : actif
- **Décision** : RTF (*real-time factor*) = temps de traitement / durée de l'audio.

### D-37 — Validation de la chaîne d'évaluation par deux sanity checks

- **Date** : 03–04/11/2026
- **Statut** : fait
- **Décision** : script `metrics_and_sanity_checks.py`.
  - Check 1 : référence vs référence, DER attendu = 0.
  - Check 2 : hypothèse « un seul locuteur », construite en mémoire avec `Annotation.from_records`.
- **Résultat observé du check 2** :
  - conditions : 40 fichiers dev (D-22), DER moderne, collar 0, UEM \[0, durée\], micro ;
  - DER = **53,66 %** (FA 10,34 %, missed 2,50 %, confusion 40,81 %) ;
  - c'est le **plancher que la baseline doit battre**.
- **Cas relevé** : pqmho.
  - 14,92 s de parole annotée, le reste est de la musique.
  - L'écoute du 04/11 confirme que les annotations sont correctes.
  - C'est le premier cas d'échec attendu pour la VAD par énergie.

### Métriques encore à confirmer

- Évaluation du clustering : purity, coverage, speaker count error. *Non décidé.*

## 5. Modèles pré-entraînés et risque de fuite

### D-40 — Règles contre la fuite de données avec les modèles pré-entraînés

- **Date** : 28/10/2026
- **Statut** : actif
- **Contexte** : `pyannote/segmentation-3.0` a été entraîné sur les ensembles d'entraînement de 9 corpus, dont AMI et VoxConverse. Les splits utilisés ne sont pas précisés.
- **Décision** :
  1. AMI ne doit être évalué que sur dev ou test d'AMI-diarization-setup, jamais sur train.
  2. VoxConverse doit être évalué de préférence sur le test v0.3, tant que les fichiers d'entraînement de pyannote ne sont pas identifiés.
  3. Pour chaque modèle pré-entraîné, il faut documenter ses corpus d'entraînement et signaler tout recouvrement possible dans le tableau de résultats.
  4. L'évaluation doit être complétée par des enregistrements maison inédits.
- **Justification** : un modèle évalué sur des données qu'il a vues donne un DER trop optimiste, ce qui biaise la comparaison avec la baseline.

## 6. Baseline classique

### D-50 — Prétraitement audio : 16 kHz mono

- **Date** : 27–28/10/2026
- **Statut** : actif
- **Décision** :
  - l'audio est chargé avec librosa, rééchantillonné à 16 kHz en mono ;
  - les enregistrements maison `.m4a` sont d'abord convertis en `.wav` avec ffmpeg.
- **Justification** : 16 kHz est la fréquence standard des modèles de parole. *Justification détaillée à compléter.*

### D-51 — Paramètres STFT de l'exploration

- **Date** : 01/11/2026
- **Statut** : provisoire
- **Décision** : trame de 64 ms, hop de 10 ms (`experimenations.py`).
- **Note** : ces paramètres ont été utilisés pour l'exploration. Leur usage dans la baseline reste à confirmer.

### D-52 — Énergie par trame : RMS calculée depuis la STFT

- **Date** : 04/11/2026
- **Statut** : actif
- **Options** :
  - `librosa.feature.rms` sur le signal ;
  - `librosa.feature.rms(S=...)` en réutilisant la STFT.
- **Décision** : `librosa.feature.rms(S=...)`. Fonction `energy_vad`, écrite par Grégoire.
- **Justification** : on réutilise la STFT déjà calculée. *À compléter.*

### D-53 — Seuil de la VAD : relatif au maximum du fichier, −23 dB

- **Date** : 04/11/2026
- **Statut** : provisoire
- **Options** : *à compléter*
- **Décision** : seuil à −23 dB sous le maximum de RMS du fichier.
- **Justification** : valeur de départ choisie à l'œil sur le fichier ysgbf (visualisation RMS + seuil + zones de référence). Elle sera affinée sur le sous-ensemble dev.
- **Revoir si** : la detection error (D-35) sur le sous-ensemble dev montre un meilleur seuil.

## 7. Organisation du travail

### D-60 — Recalage du planning : finir la baseline avant pyannote

- **Date** : 03/11/2026
- **Statut** : actif, de fait
- **Contexte** : environ 3 jours de retard sur le planning de la semaine 1.
- **Options** :
  - A : finir la baseline complète d'abord ;
  - B : faire une baseline minimale puis passer à pyannote.
- **Décision** : A, appliquée de fait.

## Décisions ouvertes

| Sujet | Options envisagées |
| --- | --- |
| Smoothing de la VAD | median filter / hystérésis / durées minimales |
| Segmentation et clustering de la baseline | agglomératif / spectral / k-means ; nombre de locuteurs connu ou estimé |
| Modèles de la semaine 2 | A : modèles pré-entraînés / B : fine-tuning |
| Corpus de test secondaire | A : enregistrements maison seuls / B : + Simsamu (FR) / C : + SUMM-RE (FR) |
| AMI en corpus secondaire | oui / non |
| Organisation du code | module séparé réutilisable (ex. `viz.py`), probable |
| Machine d'entraînement | PC fixe (RTX 3070, 8 Go) / Google Colab / autre |
| Métriques de clustering | purity, coverage, speaker count error |
