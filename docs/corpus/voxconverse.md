# Fiche corpus — VoxConverse

> Dernière vérification des sources : 28 octobre 2026.
> Pages officielles : https://www.robots.ox.ac.uk/~vgg/data/voxconverse/ (Oxford VGG), https://mm.kaist.ac.kr/datasets/voxconverse/ (miroir KAIST), https://github.com/joonson/voxconverse (annotations)
> Article de référence : Chung et al., « Spot the conversation: speaker diarisation in the wild », Interspeech 2020 (https://arxiv.org/abs/2007.01216)

## Légende de fiabilité

| Étiquette | Signification |
|---|---|
| **[OFFICIEL]** | Vérifié sur une page officielle, le dépôt des auteurs, ou un article des auteurs |
| **[TIERS-PUBLIÉ]** | Article scientifique ou dépôt technique reconnu, écrit par d'autres que les auteurs du corpus |
| **[COMMUNAUTÉ]** | Fiche Hugging Face, liste GitHub communautaire, blog ou forum. À prendre avec prudence |
| **[À VÉRIFIER]** | Information non trouvée, contradictoire, ou hypothèse de notre part |

Note : la page d'Oxford n'a pas pu être lue directement par notre outil (accès bloqué par le site). Les mêmes informations ont été lues sur le README GitHub des auteurs et sur le miroir KAIST.

## En une phrase

Environ 64 h d'extraits de vidéos YouTube : débats politiques, plateaux TV, journaux, talk-shows. On y trouve de 1 à 21 locuteurs par fichier, avec des annotations RTTM fournies directement et produites par une chaîne semi-automatique audio-visuelle vérifiée par des humains. **[OFFICIEL]**

---

## 1. Accès

- **Audio en libre téléchargement**, sans inscription mentionnée, sous forme de deux archives zip de fichiers WAV : `voxconverse_dev_wav.zip` et `voxconverse_test_wav.zip`. Des sommes MD5 sont fournies pour vérifier l'intégrité. **[OFFICIEL]**
- **Annotations au format RTTM, directement**, dans le dépôt GitHub `joonson/voxconverse`, dossiers `dev/` et `test/`. **Aucune conversion n'est nécessaire.** **[OFFICIEL]**
- **Versions des annotations** **[OFFICIEL]** :
  - **v0.3**, sur la branche `master` : corrige des erreurs découvertes dans certains RTTM du test. C'est la version recommandée par les auteurs.
  - **v0.2**, sur la branche `ver0.2` : version antérieure.
  - Il faut noter dans notre protocole la version utilisée (v0.3). C'est aussi celle utilisée par pyannote 3.1 dans ses benchmarks.
- **Taille** : environ 2,3 Go pour le dev et 5,0 Go pour le test, selon la version parquet Hugging Face `diarizers-community/voxconverse`. **[COMMUNAUTÉ]**
  - La taille exacte des zip officiels n'a pas été trouvée. **[À VÉRIFIER]**
  - Les zip sont *a priori* à télécharger en entier, même si on ne garde ensuite que 10 à 15 fichiers. **[À VÉRIFIER]**
- **Pas de vidéo** : seul l'audio est publié. La partie visuelle n'a pas encore été diffusée. **[COMMUNAUTÉ]** (AwesomeDiarizationDataset) Pour nous ce n'est pas un problème, puisqu'on travaille en audio seul.

## 2. Licence

- **CC BY 4.0, « for research use »**. **[OFFICIEL]** (README GitHub, page KAIST)
- **Le droit d'auteur de chaque vidéo reste à son propriétaire d'origine** (chaînes YouTube, médias). **[OFFICIEL]**
  - Notre interprétation : c'est une zone plus floue qu'AMI. La licence CC BY 4.0 couvre le travail des auteurs (sélection, annotations), mais pas forcément le contenu audio lui-même.
  - Dans un projet académique sans redistribution, le risque pratique est faible. **Il ne faut en revanche pas mettre l'audio dans le dépôt GitHub** (déjà prévu via `.gitignore`).
  - Ce point mérite d'être mentionné dans le rapport. **[À VÉRIFIER si on veut aller plus loin]**
- **Consentement** : les personnes filmées sont des personnalités publiques dans des vidéos publiques. Il n'y a **pas de formulaire de consentement** comme pour AMI. Les auteurs avertissent que la distribution des identités peut ne pas représenter la population mondiale, avec des biais possibles de genre, d'origine, de langue, etc. **[OFFICIEL]**

## 3. Langue

- **Information contradictoire** :
  - La fiche Hugging Face `diarizers-community/voxconverse` indique « English ». **[COMMUNAUTÉ]**
  - L'article *Benchmarking Diarization Models* (2025) le décrit comme « An English language conversation dataset ». **[TIERS-PUBLIÉ]**
  - La liste communautaire `AwesomeDiarizationDataset` indique « Multi », c'est-à-dire multilingue. **[COMMUNAUTÉ]**
  - **Ni l'article original ni les pages officielles ne précisent la langue.** **[OFFICIEL : non renseigné]**
- Notre hypothèse : le corpus est majoritairement en anglais, avec peut-être quelques extraits dans d'autres langues. **[À VÉRIFIER par écoute]** C'est une bonne chose à noter dans la section « Observations personnelles ».

## 4. Volume

| Élément | Dev | Test | Fiabilité |
|---|---|---|---|
| Nombre de fichiers | 216 | 232 | [OFFICIEL] (article 2020 + VoxSRC 2021) |
| Durée | environ 20,3 h | environ 43,5 h | [OFFICIEL] (Table 2 de l'article) |
| Locuteurs par fichier (min / moyenne / max) | 1 / 4,5 / 20 | 1 / 6,5 / 21 | [OFFICIEL] |
| Durée d'un fichier (min / moyenne / max) | 22 s / 5,6 min / 18 min | 26 s / 11,3 min / 20 min | [OFFICIEL] |
| Proportion de parole (moyenne) | 93,2 % | 89,6 % | [OFFICIEL] |
| Chevauchement (moyenne, max) | 3,8 %, max 28,7 % | 3,1 %, max 29,8 % | [OFFICIEL] |

- **Total : 448 fichiers, environ 64 h.** Les auteurs l'ont confirmé lors du challenge VoxSRC 2021 (3 830 min). **[OFFICIEL]**
- **Chiffres incohérents selon les sources** :
  - L'article de 2020 parle de « 70 heures » dans sa conclusion.
  - La page KAIST dit « plus de 50 heures ».
  - Le rapport VoxSRC 2020 mentionne un test de 310 fichiers et 56 h, qui correspond probablement à une version antérieure à la publication.

  On retient **448 fichiers / environ 64 h pour la version v0.3**. **[À VÉRIFIER : recompter après téléchargement]**
- Les splits « dev » et « test » sont les noms donnés par les auteurs. Le split dev/validation de notre projet sera construit à partir de ces fichiers et documenté séparément.

## 5. Conditions d'enregistrement

- **Vidéos YouTube « dans la nature »** : débats politiques, tables rondes, interviews de célébrités, sketches d'actualité, talk-shows. Elles ont été choisies volontairement pour contenir de la parole superposée. **[OFFICIEL]**
- **Conditions variées et non contrôlées** : bruit de fond, musique, rires, applaudissements, distances au micro variables, montage TV. **[OFFICIEL]** (« challenging background conditions ») + **[TIERS-PUBLIÉ]** (« background noise, varying microphone distances, and uncontrolled recording environments »)
- **Un seul canal audio par vidéo**, déjà mixé par le média. Il n'existe pas de version « propre » comme les micros-casques d'AMI. [Interprétation]
- Repère chiffré : pyannote 3.1 obtient un DER de **11,3 %** sur VoxConverse v0.3, sans collar et avec les chevauchements comptés. **[TIERS-PUBLIÉ]** C'est plus facile qu'AMI en SDM (22,4 %) avec le même protocole, malgré le bruit, probablement grâce au taux de chevauchement plus faible. [Interprétation]

## 6. Qualité des annotations (selon les auteurs et la littérature)

### Méthode d'annotation (article de 2020) **[OFFICIEL]**

Les annotations sont **semi-automatiques**. Une chaîne automatique audio-visuelle propose les étiquettes, et des humains les vérifient et les corrigent :

1. Détection des plans, puis détection et suivi des visages.
2. Regroupement des visages par identité.
3. **Détection du locuteur actif**, c'est-à-dire de qui bouge les lèvres en synchronisation avec le son. Deux méthodes sont combinées, et une piste n'est retenue que si les deux sont d'accord, ce qui réduit les fausses alarmes dues aux rires et à la musique.
4. **Locuteurs hors-champ** : la parole sans visage visible est attribuée par comparaison d'empreintes vocales (embeddings). En cas de doute, la décision revient à un annotateur humain.
5. **Vérification et correction manuelles** avec l'outil VIA. Pour le test, les annotations ont été vérifiées manuellement avant publication, avec l'aide d'une société d'annotation (Elancer). **[OFFICIEL]** (VoxSRC 2020/2021)

### Règles d'annotation **[OFFICIEL]**

- Un segment est coupé dès qu'il y a une **pause de plus de 0,25 s**.
- Les annotateurs doivent placer les frontières **à 0,1 s près** de la vraie frontière.
- **Les rires ne sont attribués à personne.**
- Les mots courts comme « yes » comptent comme parole s'ils sont transcriptibles.

### Mesure de la qualité **[OFFICIEL]**

- Une heure de contenu (15 vidéos) a été annotée **indépendamment par deux annotateurs**. Leur désaccord mesuré en DER est d'**environ 1 %**, principalement sur l'attribution des locuteurs hors-champ.
- La version v0.3 corrige des erreurs découvertes après coup dans certains RTTM du test, ce qui montre que des erreurs ont existé.

### Points d'attention relevés par d'autres

- L'article arXiv 2507.09226 (2025) classe VoxConverse parmi les corpus **« conçus pour la diarisation »**, avec des frontières précises (moins de 100 ms, découpage aux pauses de plus de 250 ms), contrairement à AMI. **[TIERS-PUBLIÉ]**

### Faiblesses connues de la chaîne d'annotation (selon les auteurs) **[OFFICIEL]**

- Le détecteur de locuteur actif peut se déclencher à tort si un son de fond « ressemble » au mouvement des lèvres d'une personne visible.
- Les erreurs de regroupement d'identités causent des confusions de locuteurs, la principale source d'erreur.
- **L'attribution des locuteurs hors-champ** est la principale source de désaccord entre annotateurs.

### Risque de fuite de données (lien avec le split de mercredi)

- **Hypothèse** : des personnalités publiques (des hommes politiques par exemple) pourraient apparaître dans plusieurs vidéos différentes. Un même locuteur pourrait alors se retrouver à la fois dans notre dev et notre validation. **[À VÉRIFIER]**
- Ce risque n'est mentionné dans aucune source trouvée.
- Les étiquettes de locuteurs des RTTM sont *a priori* propres à chaque fichier (`spk00`, `spk01`…), ce qui ne permet pas de détecter facilement un même locuteur entre fichiers. **[À VÉRIFIER en ouvrant un RTTM]**
- Pour une approche non supervisée (clustering), ce risque est faible. Il deviendrait important si on entraîne ou adapte un modèle. [Interprétation]

## 7. Rumeurs et sources communautaires

Les recherches dans les forums (Reddit, issues GitHub) n'ont **rien donné de substantiel** sur des problèmes d'annotation de VoxConverse au-delà de la correction v0.3 documentée par les auteurs. Les points suivants viennent de sources non officielles :

- **Langue** : contradiction entre « English » (Hugging Face) et « Multi » (AwesomeDiarizationDataset), cf. §3. **[COMMUNAUTÉ]**
- La fiche Hugging Face `diarizers-community/voxconverse` affiche dans son exemple des splits 136/18/16, identiques à ceux d'AMI, alors que ses métadonnées indiquent 216 / 232. **[COMMUNAUTÉ]** Notre interprétation est qu'il s'agit d'un copier-coller entre fiches, et qu'il faut se fier aux métadonnées.
- Un taux de chevauchement de 3,28 % est cité par AwesomeDiarizationDataset, cohérent avec l'article (3 à 4 %). **[COMMUNAUTÉ]**

## 8. Limites identifiées

- **Domaine** : uniquement des médias TV/YouTube, avec un montage professionnel. Ce n'est pas la même chose qu'une conversation enregistrée au téléphone ou en réunion. [Interprétation]
- **Chevauchement faible en moyenne** (3 à 4 %) mais très variable (jusqu'à environ 30 % sur certains fichiers). C'est moins représentatif des réunions animées. [Interprétation]
- **Nombre de locuteurs très variable** (1 à 21) : intéressant pour tester l'estimation du nombre de locuteurs, mais difficile pour une baseline classique. [Interprétation]
- **Fichiers longs** dans le test (jusqu'à 20 min) : temps de calcul plus élevé.
- **Droits sur l'audio** moins clairs qu'AMI (cf. §2).
- Annotations **semi-automatiques**, avec des erreurs possibles sur les locuteurs hors-champ.
- **Langue non documentée officiellement.**

## 9. Observations personnelles

> Section à compléter par Grégoire après téléchargement et écoute de quelques fichiers.

- Fichiers téléchargés (ID, split) :
- Langue(s) entendue(s) :
- Écoute : la parole est-elle bien couverte par les RTTM ? Les frontières tombent-elles au bon moment ?
- Rires, musique, applaudissements : comment sont-ils traités dans les RTTM ?
- Même personnalité dans plusieurs fichiers ?
- Difficultés rencontrées (téléchargement, format, scripts) :
- Autres remarques :

---

## Sources

- README GitHub des auteurs (versions, licence, liens) : https://github.com/joonson/voxconverse (lu via https://raw.githubusercontent.com/joonson/voxconverse/master/README.md)
- Page miroir KAIST : https://mm.kaist.ac.kr/datasets/voxconverse/
- Page officielle Oxford VGG (non lue directement, accès bloqué) : https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- Chung et al., « Spot the conversation: speaker diarisation in the wild », Interspeech 2020 : https://arxiv.org/pdf/2007.01216
- VoxSRC 2020 : https://arxiv.org/pdf/2012.06867
- VoxSRC 2021 : https://arxiv.org/pdf/2201.04583
- Fiche pyannote/speaker-diarization-3.1 : https://huggingface.co/pyannote/speaker-diarization-3.1
- Lanzendörfer et al., « Benchmarking Diarization Models » (2025) : https://arxiv.org/html/2509.26177v1
- « Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization? » (2025) : https://arxiv.org/pdf/2507.09226
- Hugging Face diarizers-community/voxconverse : https://www.huggingface.co/datasets/diarizers-community/voxconverse/blob/main/README.md
- AwesomeDiarizationDataset : https://github.com/liutaocode/AwesomeDiarizationDataset
- VoxMM (KAIST, 2024) : https://mm.kaist.ac.kr/pubs/pdfs/kwak24a.pdf
