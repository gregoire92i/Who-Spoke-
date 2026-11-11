# Notes importantes et difficultés

## problemes rencontrés lors de la découverte du corpus (28 octobre):
La fiche officielle de pyannote/segmentation-3.0 indique un entraînement sur neuf corpus, dont AMI et VoxConverse. Elle ne précise pas quelles parties de chaque corpus ont été utilisées.

Conclusion : Pour la semaine 2 faire attention si le choix de faire du finetuning est fait -> n'utiliser ni AMI ni Voxconverse et verifier le corpus utilisé

Trois règles qu’on se fixe :
- Pour AMI, n’évaluer que sur des réunions dev ou test, jamais train. Pour VoxConverse, privilégier le test v0.3.
- Pour chaque modèle pré-entraîné, noter les corpus sur lesquels il a été entraîné et signaler tout recouvrement possible avec les fichiers d’évaluation.
Compléter l’évaluation par des enregistrements maison jamais publiés.

## Difficultés rencontrées : téléchargement de VoxConverse (28-29 oct. 2026)

### Téléchargement long et instable
- Le site d'Oxford n'étant pas disponible pour l'audio, passage par le miroir KAIST.
- Le serveur coupe la connexion environ tous les 1 Go : il a fallu relancer `wget -c` (reprise) à de nombreuses reprises.
- Au total, environ 4 h pour récupérer les deux archives (dev + test).
- À retenir pour la reproduction : prévoir du temps, utiliser `wget -c`, et vérifier les sommes MD5 après téléchargement.

### Doublons dans le dossier `test`
- Constat : `find data/voxconverse/test -type f -name "*.wav" | wc -l` donne **464** fichiers, alors que le test contient **232** fichiers selon les auteurs (le dev est correct : 216).
- hypothese de départ: probable mauvaise manipulation de ma part (double dezippage)

- Cause trouvée : l'archive de test contient un dossier `__MACOSX/` (métadonnées ajoutées quand un zip est créé sur Mac). Il contient 232 fichiers `._nom.wav` qui ne sont pas des enregistrements, mais que `find -name "*.wav"` compte. 232 vrais fichiers (dans `voxconverse_test_wav/`) + 232 faux = 464.
- Hypothèse écartée : un double dézippage

## Réparition dev/test sur voxconverse

Les deux ensembles ne se ressemblent pas tout à fait (chiffres des auteurs). Le test a plus de locuteurs par fichier (6,5 contre 4,5 en moyenne) et des fichiers deux fois plus longs (11,3 min contre 5,6 min)

### conclusions de data vizualization 

Quelques outliers, on verra ce qu'on en fait après avoir fini d'établir la baseline
- **Locuteurs récurrents (VoxConverse)** : aucune source (article Chung et al. 2020, rapports VoxSRC 2020/2021, rétrospective 2024) ne dit si une même personne apparaît dans plusieurs fichiers, ni entre dev et test. Les étiquettes RTTM v0.3 sont locales à chaque fichier (vérifié : 20 étiquettes distinctes dans tout le dev, 21 dans le test), et les auteurs ont dédoublonné les vidéos, pas les personnes. Question ouverte, vérifiable seulement par comparaison d'embeddings entre fichiers. Impact négligeable pour la baseline (aucun entraînement) ; à vérifier avant tout fine-tuning sur le dev en semaine 2.

## Décisions d'évaluation
Pour l'instant :
on fixe le collar=0ms convention NIST -> eviter de penaliser les imprecisions de notation -> porbleme : maj du collar à 0 pour pouvoir comparzer aux modeles pyannote
- DER qui compte les overlap (approche moderne) -> metrique principale
- DER sans les overlap (approche ancienne "forgiving")
- JER
|-> comparer les deux -> pour la baseline surtout.
UEM = [0, durée réelle de l'audio]
agrégation de plusieurs fichiers -> micro, car plus représentatif.

### Metriques pour la detection de parole seule

- detection error=(FA+MISS)/speaking duration

### Evaluation du clustering
A cofnirmer plus tard : 
- Purity ->Chaque cluster du système contient-il un seul vrai locuteur ? - equivalent to accuracy
- Coverage -> Chaque vrai locuteur est-il rassemblé dans un seul cluster ? -> equivalent to recall ? -> CHECK - OK
- Speaker count error -> Écart entre nombre de locuteurs prédit et réel

### Mesure des Couts de calcul
- RTF (Real-Time Factor) = temps de traitement / durée de l'audio -> si =1 -> audio traité aussi vite que sa durée reelle 
### Sanity checks
- ref vs ref
- one speaker partout -> DER de reference

## Problemes de RAM
tentative de faire tourner la baseline sur tous les fichiers en meme temps -> provoque un bug wsl car manque de RAM
-> solution : faire passer les fichiers 1 par 1.

## DER horrible sur la baseline (50%)
-> voir rapport intermediaire -> problème vient surement des embeddings car le clustering spectral est instable et se base surtout sur les embeddings
-> test -> doubler les windods d'embedding -> der modern passe de 43% sur le dev à 33%  et 31% pour le der forgiving???? incroyable
-> le cle du clustering est l'embedding, on va partir sur un reglage methodique des frames et du hop qui vont ensuite partir pour les windows d'embedding pour avoir des bases fiables.

### Frames et hop, leur impact sur le DER ?
-> (voir graphiques) -> une tendance se dessine : 
1 le hop ne semble pas avoir d'impact sur le DER 
2 : le der diminue très legerement quand la taille des frames augmente

## MODE ORACLE : VAD, eigeingap, et post treatment
### VAD
-> VAD en mode oracle (vad parfaite)-> nous fait gagner ~4% sur le DER
***PROBLEMES*** : 
- Parole VAD sans label : 3.33% -> frames que la VAD a classées en parole, mais qu’aucune window ne couvre -> moment ou qqun parle mais aucun locuteur n'y est attribué -> pourquoi ?
- Parole départagée     : 4.17% -> Ce sont les frames où les windows qui les couvrent sont en désaccord à égalité. C’est le score au centroïde qui tranche (n_tie).

***CONCLUSION*** : La VAD n'est pas un gros facteur limitant, à voir comme amélioration secondaire, mais en departageant simplement les 3.33% (ou 2.24% pour la vad pas en mode oracle), on peut gagner simplement

### Eigeingap
-> Eigeingap en mode oracle : nombre de locuteurs connus
***PROBLEME*** : en mode oracle Eigeingap ou on connait le nombre de locuteurs, le DER est PLUS ELEVEE DE 5% que sans (33 à 38%) -> confusion passe de 21.5% à 26.8% -> clustering et embedding coupables ?
***TESTS POSSIBLES***
-> garder le clustering et remplacer les embeddings par des embeddings pré-entraînés
-> ou inversement remplacer le clustering par AHC

-> fichier problematique : [73/90] tucrg | DER 474.2% | k 3 (réf 3) | RTF 0.002
-> pour la baseline normal on a 2.24% de parole sans label et 6.52% de parole départagée. Ici, on a 2,24 % et 17,38 %. -> etrange mais pas anormal car lele eigeingap a tendance à sous estimer le nbre de locuteurs -> donc moins de frontieres entre locuteurs sans le mode oracle

### Post treatment
Sur les 6,52 % de parole départagée, le départage par score se trompait donc sur 1,90 %, soit environ 30 % des cas. Pour le reste, soit il avait déjà choisi le bon label, soit aucune des deux windows ne portait le bon locuteur.

***Résultat du diagnostic***
Le DER moderne passe de 33,49 % à 31,63 %, soit −1,86 point.
Le false alarm et le missed sont identiques au run baseline (4,17 % et 7,77 %). Le gain vient entièrement de la confusion, comme attendu.
Le gain (1,86 point) correspond bien à la parole modifiée (1,90 %), ce qui confirme que les modifications vont dans le bon sens.