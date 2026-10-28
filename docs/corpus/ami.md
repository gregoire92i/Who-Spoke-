# Fiche corpus — AMI Meeting Corpus

> Dernière vérification des sources : 28 octobre 2026.
> Site officiel : https://groups.inf.ed.ac.uk/ami/corpus/

## Légende de fiabilité

| Étiquette | Signification |
|---|---|
| **[OFFICIEL]** | Vérifié sur le site officiel du corpus ou dans un article publié par ses auteurs |
| **[TIERS-PUBLIÉ]** | Article scientifique ou dépôt technique reconnu, écrit par d'autres que les auteurs du corpus |
| **[COMMUNAUTÉ]** | Fiche Hugging Face, liste GitHub communautaire, blog ou forum. À prendre avec prudence |
| **[À VÉRIFIER]** | Information non trouvée, contradictoire, ou hypothèse de notre part |

## En une phrase

Environ 100 h de réunions en anglais, enregistrées dans trois salles équipées, avec un micro-casque par participant et des micros d'ambiance. La plupart des réunions sont des jeux de rôle à 4 personnes. **[OFFICIEL]**

---

## 1. Accès

- **Téléchargement libre depuis le site officiel**, via une page qui permet de choisir les flux audio réunion par réunion. Les annotations sont dans deux fichiers zip (annotations manuelles et annotations automatiques). La page de téléchargement ne mentionne aucune inscription obligatoire. **[OFFICIEL]** (https://groups.inf.ed.ac.uk/ami/download)
- **Miroir OpenSLR (SLR16)** : `headset.tar.gz` (24 Go, micros-casques), `Array1-01.tar.gz` … `Array1-08.tar.gz` (environ 7,5 Go chacun), annotations `ami_manual_1.6.1.tar.gz` (21 Mo). **[TIERS-PUBLIÉ]** (https://openslr.org/16/)
- **Taille moyenne par réunion** selon le flux **[OFFICIEL]** :
  - *Headset mix*, c'est-à-dire tous les micros-casques mixés en un seul WAV : environ 30 Mo.
  - *Individual headsets*, 4 fichiers WAV (un par participant) : environ 120 Mo.
  - *Microphone array*, deux réseaux de micros posés sur la table : environ 360 Mo.

  Pour nous, le *headset mix* d'une dizaine de réunions représente environ 300 Mo, ce qui reste raisonnable.
- **Format des annotations d'origine** : format NXT (XML), pas en RTTM. **[OFFICIEL]**
  - **Solution existante** : le dépôt `BUTSpeechFIT/AMI-diarization-setup`, et son fork `pyannote/AMI-diarization-setup`, fournit des **fichiers RTTM et UEM prêts à l'emploi**, dérivés des annotations manuelles v1.6.2. Il fournit aussi des listes train/dev/test. **[TIERS-PUBLIÉ]**
  - *UEM* : fichier qui indique quelles portions de l'audio doivent être évaluées.
  - Le fork pyannote contient un script de téléchargement de l'audio. **[TIERS-PUBLIÉ]** (blog de H. Bredin, auteur de pyannote : https://herve.niderb.fr/posts/2022-12-02-how-I-won-2022-diarization-challenges.html)
- **Version Hugging Face prétraitée** : `diarizers-community/ami`, au format parquet, avec deux sous-ensembles `ihm` et `sdm`, 20,9 Go au total. **[COMMUNAUTÉ]** La fiche est incohérente sur le nombre de lignes par split.

## 2. Licence

- **Site officiel : CC BY 4.0** pour les signaux, la transcription et une partie des annotations. Usage libre, y compris commercial, avec obligation de citer la source, de mettre un lien vers la licence et d'indiquer les modifications. **[OFFICIEL]** (https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- **Contradiction avec des sources plus anciennes** :
  - OpenSLR indique « CC BY-NC-SA 2.0 (modified) ». **[TIERS-PUBLIÉ]**
  - Un article de 2007 des auteurs parle d'une licence « basée sur CC BY-NC-SA 2.5 ». **[OFFICIEL, mais ancien]**

  Notre interprétation est que la licence a été assouplie depuis, et c'est la page officielle actuelle qui fait foi. **[À VÉRIFIER]** : dater ce changement si on veut être rigoureux dans le rapport.
- **Consentement** : les participants ont signé un formulaire de consentement, dont une copie vierge est disponible sur la page « Ethics and consent ». Le contrat a fait l'objet d'une revue éthique par la Commission européenne. **[OFFICIEL]** Le contenu exact du formulaire n'a pas été lu. **[À VÉRIFIER]**
- La licence CC BY 4.0 ne couvre pas les droits de la personnalité, comme la vie privée ou le droit à l'image. **[OFFICIEL]** Pour nous, la conséquence pratique est de ne pas redistribuer l'audio dans le dépôt GitHub (déjà prévu via `.gitignore`).
- Les dépôts `AMI-diarization-setup` (RTTM) sont sous licence Apache-2.0. **[TIERS-PUBLIÉ]**

## 3. Langue

- **Anglais**, avec une **majorité de locuteurs non natifs**. **[OFFICIEL]** (page Overview)
- Certaines réunions non scénarisées ont des locuteurs à **fort accent français** : IN1001, IN1008, IN1009. **[OFFICIEL]**

## 4. Volume

| Élément | Valeur | Fiabilité |
|---|---|---|
| Durée totale | environ 100 h | [OFFICIEL] |
| Réunions scénarisées | environ 2/3 du corpus, 4 participants jouant une équipe de conception (une série = 4 réunions a, b, c, d d'environ 1 h maximum) | [OFFICIEL] |
| Réunions non scénarisées | environ 1/3, 3 à 5 participants selon la réunion | [OFFICIEL] |
| Locuteurs par réunion | 3 à 5, le plus souvent 4 | [OFFICIEL] + [COMMUNAUTÉ] |
| Partition « Full-corpus-ASR » (utilisée par AMI-diarization-setup) | train environ 81 h ; dev = 9 séries ; test = 4 séries | [OFFICIEL] |
| Nombre de réunions par split dans cette partition | 136 / 18 / 16 (train / dev / test) | [COMMUNAUTÉ], cohérent avec notre comptage des séries officielles |
| Taux de chevauchement de parole | environ 13 à 16 % selon la source et le sous-ensemble | [TIERS-PUBLIÉ] (16 % sur dev+test, arXiv 2509.26177) ; [COMMUNAUTÉ] (13,4 %, AwesomeDiarizationDataset) |

Pour nous, 10 à 15 réunions représentent environ 10 h d'audio, ce qui est largement suffisant pour une baseline.

## 5. Conditions d'enregistrement

- **Trois salles instrumentées** aux propriétés acoustiques différentes : Édimbourg, IDIAP (Suisse) et TNO (Pays-Bas). **[OFFICIEL]**
- **Micros par salle** : un micro-casque par participant, un réseau circulaire de 8 micros, plus des micros-cravates. Il y a aussi 6 caméras et des captures du tableau blanc, des slides et des stylos. **[OFFICIEL]**
- **Plusieurs façons d'écouter la même réunion**, ce qui change la difficulté :
  - **IHM / *Headset mix*** (*Individual Headset Microphones*) : son proche et propre. C'est le cas le plus facile.
  - **SDM** (*Single Distant Microphone*, par exemple `Array1-01`) : un seul micro posé sur la table, avec de la réverbération et des voix plus faibles. C'est le cas le plus réaliste et le plus difficile.
  - Repère chiffré : pyannote 3.1 obtient un DER de **18,8 %** en headset mix et **22,4 %** en SDM, sans collar et avec les chevauchements comptés. **[TIERS-PUBLIÉ]** (https://huggingface.co/pyannote/speaker-diarization-3.1)
- Les conditions sont **contrôlées** : salle de réunion et participants assis. Elles sont donc moins variées que des vidéos « dans la nature ». **[TIERS-PUBLIÉ]** (VoxMM, KAIST : AMI est « recorded in a controlled environment »)
- **Particularités signalées par les auteurs** :
  - EN2006a : participants mal installés.
  - IN1014 : l'enregistrement s'arrête avant la fin de la réunion.
  - EN2004a : uniquement des locutrices.

  **[OFFICIEL]**

## 6. Qualité des annotations (selon les auteurs et la littérature)

### Ce que disent les auteurs

- **Transcription manuelle mot à mot**, réalisée avec l'outil ChannelTrans, en 2 voire 3 passes. **[OFFICIEL]**
  1. Un pré-segmenteur automatique basé sur l'énergie détecte la parole sur chaque micro-casque.
  2. Les annotateurs ajustent les frontières pendant la première passe.
  3. Une deuxième passe relit tous les segments et ajoute la parole manquée.
  4. Un script de validation recherche les erreurs.
- **Les instants de début et de fin des mots** sont obtenus **automatiquement par alignement forcé**, et non placés à la main. **[OFFICIEL]** L'alignement forcé consiste à faire caler par un modèle de reconnaissance vocale chaque mot de la transcription sur le signal.
- Les releases jusqu'à la 1.4 contenaient des erreurs dans les fichiers NXT, corrigées ensuite. **[OFFICIEL]** Les RTTM d'AMI-diarization-setup utilisent la version 1.6.2.

### Points d'attention relevés par d'autres

- **Deux variantes de référence** dans AMI-diarization-setup :
  - `only_words` : seuls les mots sont comptés comme parole.
  - `word_and_vocalsounds` : les mots plus les rires, toux, etc.

  Les auteurs du setup **recommandent `only_words`** parce que les annotateurs ont marqué les sons vocaux de façon incohérente. **[TIERS-PUBLIÉ]**
- **Frontières « larges »** : un article de 2025 (arXiv 2507.09226) soutient que les corpus conçus pour la reconnaissance vocale, dont AMI, ont des segments qui **incluent souvent des pauses internes**. **[TIERS-PUBLIÉ]**
  - En comparant les étiquettes d'origine d'AMI à des étiquettes réalignées finement (pauses de plus de 200 ms coupées), ils mesurent un **écart d'environ 25 % de DER**, surtout de la parole manquée.
  - Le collar ne corrige presque rien, parce que le problème vient des pauses à l'intérieur des segments et pas d'un décalage des bords.
  - **Interprétation** : un système parfait pourrait être pénalisé sur AMI. Il faudra en tenir compte en analysant nos erreurs. C'est un seul article et ce résultat n'est pas encore un consensus. **[À VÉRIFIER]**

### Risque de fuite de données (lien avec le split de mercredi)

- Dans les réunions scénarisées, **les mêmes 4 personnes** apparaissent dans les réunions a, b, c et d d'une même série. **[OFFICIEL]**
  - Il faut donc **découper par série** (ES2008), et pas par réunion (ES2008a).
- Les auteurs ont exclu la réunion IB4005 de la partition Full-corpus parce qu'elle « a des locuteurs en commun avec deux autres ensembles ». **[OFFICIEL]**
  - On en déduit qu'ils ont cherché à éviter les locuteurs partagés entre splits, mais **la page ne le garantit pas explicitement**. **[À VÉRIFIER]**

## 7. Rumeurs et sources communautaires

Les recherches dans les forums (Reddit, issues GitHub) n'ont **rien donné de substantiel** sur la qualité des annotations AMI pour la diarisation. Les points ci-dessous proviennent de sources communautaires et sont à prendre avec prudence :

- La fiche Hugging Face `diarizers-community/ami` affiche des splits incohérents (73/18/16 dans le visualiseur contre 136/18/16 dans l'exemple). **[COMMUNAUTÉ]**
- La liste GitHub `AwesomeDiarizationDataset` donne un taux de chevauchement de 13,37 %. Ce chiffre ne vient pas d'une source officielle. **[COMMUNAUTÉ]**
- Dans le tutoriel pyannote de H. Bredin, le sous-ensemble « mini » d'AMI-SDM (28 / 3 / 3 fichiers) donne un DER de 32,5 % avec le modèle pré-entraîné, puis 26,6 % après adaptation. **[TIERS-PUBLIÉ, blog]** C'est un ordre de grandeur utile pour savoir ce qu'est un « bon » score en SDM.

## 8. Limites identifiées

- Les conditions sont contrôlées et peu variées : 3 salles, réunions de bureau, et les deux tiers du corpus sont des jeux de rôle. Le corpus risque d'être **peu représentatif des enregistrements inconnus de la démo**. [Interprétation]
- Le nombre de locuteurs est presque toujours 3 à 5. Il est peu utile pour tester l'estimation du nombre de locuteurs sur des cas extrêmes, comme 1 ou plus de 10 personnes. [Interprétation]
- Le taux de chevauchement est élevé (environ 13 à 16 %). C'est réaliste pour une réunion, mais très difficile pour une baseline classique qui n'attribue qu'un locuteur à la fois. [Interprétation]
- Les frontières sont potentiellement larges (cf. §6).
- Il faut utiliser un dépôt tiers (AMI-diarization-setup) pour obtenir les RTTM.

## 9. Observations personnelles

> Section à compléter par Grégoire après téléchargement et écoute de quelques réunions.

- Réunions téléchargées (ID, flux) :
- Écoute : la parole est-elle bien couverte par les RTTM ? Les frontières tombent-elles au bon moment ?
- Pauses internes visibles dans la timeline ?
- Qualité audio perçue (headset mix vs SDM) :
- Difficultés rencontrées (téléchargement, format, scripts) :
- Autres remarques :

---

## Sources

- Site officiel AMI : https://groups.inf.ed.ac.uk/ami/corpus/ (Overview, Licence, Download, Transcription, Datasets, Non-scenario meetings, Ethics and consent)
- OpenSLR SLR16 : https://openslr.org/16/
- BUTSpeechFIT/AMI-diarization-setup : https://github.com/BUTSpeechFIT/AMI-diarization-setup
- pyannote/AMI-diarization-setup : https://github.com/pyannote/AMI-diarization-setup
- Renals et al., ASRU 2007 (description du corpus) : https://www.cstr.inf.ed.ac.uk/downloads/publications/2007/ami-asru2007.pdf
- « Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization? » (2025) : https://arxiv.org/pdf/2507.09226
- Lanzendörfer et al., « Benchmarking Diarization Models » (2025) : https://arxiv.org/html/2509.26177v1
- Fiche pyannote/speaker-diarization-3.1 : https://huggingface.co/pyannote/speaker-diarization-3.1
- H. Bredin, blog pyannote (2022) : https://herve.niderb.fr/posts/2022-12-02-how-I-won-2022-diarization-challenges.html
- Hugging Face diarizers-community/ami : https://huggingface.co/datasets/diarizers-community/ami
- AwesomeDiarizationDataset : https://github.com/liutaocode/AwesomeDiarizationDataset
- VoxMM (KAIST, 2024) : https://mm.kaist.ac.kr/pubs/pdfs/kwak24a.pdf
