# Décision — Choix du corpus principal

> Date : 28 octobre 2026
> Statut : **décidé**, révisable si les observations de mercredi (téléchargement, écoute, timelines) révèlent un problème bloquant
> Fiches détaillées : [`corpus/ami.md`](corpus/ami.md), [`corpus/voxconverse.md`](corpus/voxconverse.md). Les étiquettes de fiabilité ([OFFICIEL], [À VÉRIFIER]…) y sont détaillées.

## Décision

**Corpus principal retenu : VoxConverse (annotations v0.3).**

Ce corpus servira à construire notre découpage dev / validation, à régler la baseline et à comparer les approches.

## Options comparées

1. **AMI Meeting Corpus** : environ 100 h de réunions en anglais, enregistrées dans 3 salles équipées.
2. **VoxConverse** : environ 64 h d'extraits de vidéos YouTube (débats, journaux, talk-shows).

## Tableau comparatif

| Critère | AMI | VoxConverse |
|---|---|---|
| **Accès** | Libre, choix réunion par réunion et flux par flux. RTTM non fournis par les auteurs : il faut passer par le dépôt tiers AMI-diarization-setup | Libre. Deux archives zip (dev, test) avec somme MD5. **RTTM fournis directement** par les auteurs (v0.3) |
| **Licence** | CC BY 4.0 (page officielle actuelle) ; licence non commerciale dans des sources plus anciennes. Participants avec formulaire de consentement | CC BY 4.0 « for research use » ; **droit d'auteur des vidéos conservé par leurs propriétaires**. Pas de consentement individuel (personnalités publiques) |
| **Langue** | Anglais, majorité de locuteurs non natifs | Majoritairement anglais selon des sources secondaires ; **non précisé officiellement** |
| **Volume** | Environ 100 h ; splits train / dev / test disponibles (136 / 18 / 16 réunions) | 448 fichiers, environ 64 h (dev 216, test 232) ; **pas de split train** |
| **Locuteurs par enregistrement** | 3 à 5, presque toujours 4 | 1 à 21 (moyenne 4,5 en dev, 6,5 en test) |
| **Conditions d'enregistrement** | Contrôlées : 3 salles, participants assis, micros-casques et micros de table. Deux tiers de jeux de rôle scénarisés | Non contrôlées : bruit, musique, rires, montage TV, distances au micro variables. **Non scénarisé** |
| **Chevauchement de parole** | Élevé : environ 16 % (étude tierce, même méthode que pour VoxConverse) | Faible en comparaison : 3,1 à 3,8 % du temps de parole en moyenne (auteurs), 3,5 % (même étude tierce), de 0 à environ 30 % selon les fichiers |
| **Qualité des annotations** | Transcription manuelle, mais instants des mots obtenus par alignement automatique. Frontières potentiellement « larges » (pauses incluses dans les segments, cf. arXiv 2507.09226) | Semi-automatique puis vérification humaine ; frontières à 0,1 s près, coupure aux pauses de plus de 0,25 s ; désaccord entre deux annotateurs d'environ 1 % de DER |
| **Répétition des locuteurs** | **Structurelle au sein d'une série** : une même équipe de 4 mène les 4 réunions a, b, c, d d'une série scénarisée (fortement probable, non écrit explicitement, à vérifier dans les RTTM). **Entre séries** : groupes différents (≈ 30 répétitions du scénario), un recouvrement avéré (IB4005), le reste non documenté | Possible pour des personnalités publiques, **non documentée** [À VÉRIFIER] |
| **Recouvrement avec l'entraînement de pyannote** | Oui (corpus listé), splits non précisés | Oui (corpus listé), **fichiers utilisés inconnus** |

## Avantages et inconvénients de chaque corpus, l'un par rapport à l'autre

### VoxConverse par rapport à AMI

**Avantages**
- **Variété** : sources, émissions, ambiances sonores et nombre de locuteurs (1 à 21) bien plus diversifiés. C'est plus proche d'enregistrements inconnus comme ceux de la démo finale.
- **Conditions réelles, non scénarisées**, contre des jeux de rôle en salle contrôlée.
- **Pas de répétition structurelle des locuteurs** entre enregistrements, contrairement aux séries d'AMI. Le risque de fuite lors du découpage dev / validation est a priori plus faible, mais pas nul (personnalités publiques récurrentes, non vérifié).
- **RTTM fournis directement par les auteurs** : aucune conversion ni dépôt tiers.
- **Frontières plus précises**, avec des règles d'annotation pensées pour la diarisation.
- Taille raisonnable des fichiers du dev (5,6 min en moyenne), ce qui est pratique pour itérer vite.

**Inconvénients**
- **Droits sur l'audio moins clairs** : le droit d'auteur des vidéos reste à leurs propriétaires. Pas de redistribution, à mentionner dans le rapport.
- **Langue non documentée officiellement.**
- **Pas de split train** : gênant seulement si on fait du fine-tuning en semaine 3.
- **Annotations semi-automatiques**, avec des erreurs connues sur les locuteurs hors champ ; une correction v0.3 a déjà été nécessaire.
- **Chevauchement faible en moyenne** : teste moins les réunions animées où plusieurs personnes parlent en même temps.
- **Fichiers d'entraînement de pyannote inconnus** : plus difficile de garantir une comparaison équitable qu'avec AMI.
- Archives à télécharger en entier (plusieurs Go) même si on n'utilise que 10 à 15 fichiers.

### AMI par rapport à VoxConverse

**Avantages**
- **Consentement explicite** des participants et licence plus claire sur l'audio.
- **Plusieurs versions audio de la même réunion** (micros-casques, micro de table) : on peut mesurer l'effet de la qualité du micro, utile pour une ablation.
- **Splits train / dev / test officiels**, utiles en cas de fine-tuning.
- **Chevauchement élevé**, représentatif des réunions réelles.
- Téléchargement fichier par fichier, donc plus léger pour un petit sous-ensemble.
- Corpus de référence très utilisé, avec de nombreux résultats publiés.

**Inconvénients**
- **Mêmes locuteurs dans plusieurs enregistrements au sein d'une même série** : chaque équipe de 4 enregistre 4 réunions, ce qui crée un risque de fuite si on découpe par réunion et non par série. Entre séries, les groupes sont différents (un recouvrement avéré : IB4005).
- **Peu de variété** : 3 salles, réunions de bureau, deux tiers de jeux de rôle, presque toujours 4 personnes. Risque de mal généraliser à la démo.
- **Pas de RTTM officiels** : dépendance à un dépôt tiers.
- **Frontières d'annotation potentiellement larges**, ce qui peut gonfler artificiellement la parole manquée dans le DER.
- Chevauchement élevé : très difficile pour une baseline classique qui n'attribue qu'un locuteur à la fois.

## Justification du choix

1. **Variété des enregistrements.** VoxConverse est fait de vidéos prises sur Internet, non scénarisées, en conditions non maîtrisées, avec des types d'enregistrements plus variés. Le système sera jugé sur des enregistrements inconnus préparés par l'organisation : un corpus varié nous donne une mesure plus réaliste de sa capacité à généraliser.
2. **Risque de fuite de données dans AMI.** Dans AMI, la structure même du corpus fait que les mêmes personnes participent à plusieurs enregistrements : une équipe de 4 enregistre les 4 réunions d'une série scénarisée. C'est fortement probable d'après la description officielle, et à confirmer dans les RTTM. Un découpage dev / validation mal fait mettrait les mêmes voix des deux côtés, et nos résultats de validation seraient trop optimistes. Le corpus n'a en revanche pas été enregistré par un seul groupe : environ 30 répétitions du scénario avec des groupes différents.

## Points de vigilance

Ces points ne remettent pas en cause la décision, mais doivent apparaître dans le rapport :

- **La fuite liée aux locuteurs existe aussi, dans une moindre mesure, dans VoxConverse.** Des personnalités publiques peuvent apparaître dans plusieurs vidéos. À vérifier en écoutant les fichiers et en lisant les RTTM.
- **Côté AMI, ce risque était maîtrisable** en découpant par série plutôt que par réunion. Le choix de VoxConverse repose donc surtout sur la variété ; l'argument de fuite est un argument complémentaire.
- **Fuite avec les modèles pré-entraînés.** pyannote/segmentation-3.0 a été entraîné sur une partie inconnue de VoxConverse. Ce risque est plus fort qu'avec AMI, dont les splits sont mieux identifiés.
  - Pour les modèles pré-entraînés, privilégier le test v0.3 pour l'évaluation.
  - Documenter les corpus d'entraînement de chaque modèle.
  - Compléter par des enregistrements maison inédits.
- **Effet du découpage sur la baseline.** Pour une baseline non supervisée (MFCC et clustering), la présence de mêmes locuteurs dans dev et validation pèse surtout via le réglage des seuils. Elle deviendrait critique si on entraîne ou adapte un modèle.

## Conséquences pour la suite

- **Mercredi 29** :
  - Télécharger les archives VoxConverse, vérifier les sommes MD5 et récupérer les RTTM v0.3.
  - Sélectionner 10 à 15 fichiers et construire notre découpage dev / validation **par fichier**, en documentant l'origine de chaque fichier (dev ou test des auteurs).
- **Protocole d'évaluation** : version des annotations (v0.3), collar et traitement des chevauchements à fixer et à noter, de manière identique pour toutes les approches.
- **Fiche VoxConverse** : compléter la section « Observations personnelles » (langue entendue, qualité des frontières, locuteurs récurrents).

## Ce qui nous ferait revoir la décision

- Des annotations v0.3 manifestement fausses sur une partie importante des fichiers écoutés.
- Une proportion importante de fichiers non anglophones, si elle pose problème pour la démo.
- Un problème d'accès ou de téléchargement bloquant.

Dans ces cas, AMI reste l'alternative documentée. Utiliser AMI en corpus secondaire, par exemple pour un test de robustesse en semaine 3, est une option **non décidée à ce stade**.
