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
on  voudrait fixer le collar à 0.1 ou 0.25 -> eviter de penaliser les imprecisions de notation -> porbleme : finalement collar à 0 pour pouvoir comparzer aux modeles pyannote
- DER qui compte les overlap (approche moderne) -> metrique principale
- DER sans les overlap (approche ancienne "forgiving")
- JER
|-> comparer les deux -> pour la baseline surtout.
UEM = [0, durée réelle de l'audio]
agrégation de plusieurs fichiers -> micro, car plus représentatif.

### Metriques pour la detection de parole seule (VAD)

- detection error=(FA+MISS)/speaking duration

### evaluation des embbedings : 
- L'AUC mesure si les embeddings permettent de distinguer deux windows du même locuteur de deux windows de locuteurs différents
L'AUC est la probabilité qu'une paire « même locuteur » prise au hasard soit plus similaire qu'une paire « locuteurs différents » prise au hasard :
- 1 : séparation parfaite, toutes les paires « même locuteur » sont au-dessus ;
- 0,5 : hasard, la similarité ne dit rien sur le locuteur ;
- moins de 0,5 : c'est inversé, des locuteurs différents se ressemblent plus que le même locuteur entre ses windows.

Dans notre cas, l'AUC se calcule ainsi :

Notations

Dans un fichier, chaque paire de windows (i, j) a une similarité s(i, j) = cos(eᵢ, eⱼ), où eᵢ est l'embedding de la window i.

- P = ensemble des paires « même locuteur » (positives), de taille |P|
- N = ensemble des paires « locuteurs différents » (négatives), de taille |N|

Formule

$$
\text{AUC} = \frac{1}{|P|,|N|} \sum_{p \in P} \sum_{n \in N} \Big( \mathbb{1}[s_p > s_n] + \tfrac{1}{2},\mathbb{1}[s_p = s_n] \Big)
$$

- 𝟙[condition] vaut 1 si la condition est vraie, 0 sinon ;
- on compare chaque paire positive à chaque paire négative ;
- une égalité compte pour ½.

### Evaluation du clustering
A cofnirmer plus tard : 
- Purity ->Chaque cluster du système contient-il un seul vrai locuteur ? - equivalent to accuracy
- Coverage -> Chaque vrai locuteur est-il rassemblé dans un seul cluster ? -> equivalent to recall ? -> CHECK - OK
- Speaker count error -> Écart entre nombre de locuteurs prédit et réel
- ARI 0,0 : l’Adjusted Rand Index compare les clusters trouvés à la vérité. 1 = parfait, 0 = pas mieux que le hasard.

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

## Reglages des parametres

### Frames et hop, leur impact sur le DER ?
-> (voir graphiques) -> une tendance se dessine : 
1 le hop ne semble pas avoir d'impact sur le DER, en tout cas le bruit est trop important pour pouvoir porter un jugement dessus
2 : le der diminue très legerement quand la taille des frames augmente (optimal vers 70ms environ)

### VAD
optimal vers -30db en seuil d'entrée reglée par grid search

### embedding windows
parametre avec le plus gros impact sur le DER (voir graphique), impact directement la confusion du clustering.
optimal vers 2.0 s


## MODE ORACLE : VAD, eigeingap, et post treatment
### VAD
-> VAD en mode oracle (vad parfaite)-> nous fait gagner ~4% sur le DER
***PROBLEMES*** : 
- Parole VAD sans label : 3.33% -> frames que la VAD a classées en parole, mais qu’aucune window ne couvre -> moment ou qqun parle mais aucun locuteur n'y est attribué -> pourquoi ?
- Parole départagée     : 4.17% -> Ce sont les frames où les windows qui les couvrent sont en désaccord à égalité. C’est le score au centroïde qui tranche (n_tie).

ATTANTION : La parole occupe 92,3 % du temps dans les fichiers de réglage. Une VAD qui marquerait tout comme parole aurait donc déjà 8,3 % de FA et 0 % de MISS. La marge de gain se situe surtout du côté des fausses alarmes.

***CONCLUSION*** : La VAD n'est pas un gros facteur limitant, à voir comme amélioration secondaire, mais en departageant simplement les 3.33% (ou 2.24% pour la vad pas en mode oracle), on peut gagner simplement

Le false alarm disparaît (−4,14 points), comme attendu.
Le missed ne baisse presque pas (−0,90 point). Avec une VAD parfaite, il reste 6,87 % de parole manquée, qui vient de deux sources :
la parole sans window, qui passe de 2,24 % à 3,33 %. La VAD de la référence contient plus de segments courts que la VAD par énergie, et un segment de moins de min_s = 0,5 s ne reçoit aucune window ;
le chevauchement de voix : la baseline ne donne qu’un seul label par frame, donc le second locuteur est toujours manqué.
La confusion ne change pas (+0,2 point). Une meilleure VAD ne règle pas le problème principal.

### Eigeingap
-> Eigeingap en mode oracle : nombre de locuteurs connus
***PROBLEME*** : en mode oracle Eigeingap ou on connait le nombre de locuteurs, le DER est PLUS ELEVEE DE 5% que sans (33 à 38%) -> confusion passe de 21.5% à 26.8% -> clustering et embedding coupables ?
***TESTS POSSIBLES***
-> garder le clustering et remplacer les embeddings par des embeddings pré-entraînés
-> ou inversement remplacer le clustering par AHC

-> pour la baseline normal on a 2.24% de parole sans label et 6.52% de parole départagée. Ici, on a 2,24 % et 17,38 %. -> etrange mais pas anormal car lele eigeingap a tendance à sous estimer le nbre de locuteurs -> donc moins de frontieres entre locuteurs sans le mode oracle

### Post treatment
Sur les 6,52 % de parole départagée, le départage par score se trompait donc sur 1,90 %, soit environ 30 % des cas. Pour le reste, soit il avait déjà choisi le bon label, soit aucune des deux windows ne portait le bon locuteur.

***Résultat du diagnostic***
Le DER moderne passe de 33,49 % à 31,63 %, soit −1,86 point.
Le false alarm et le missed sont identiques au run baseline (4,17 % et 7,77 %). Le gain vient entièrement de la confusion, comme attendu.
Le gain (1,86 point) correspond bien à la parole modifiée (1,90 %), ce qui confirme que les modifications vont dans le bon sens.

### BILAN des diagnostics oracles 

Bilan des diagnostics oracle (sample 90 du dev, collar 0)
Composant remplacé par l’oracle	DER moderne	Écart vs baseline	False alarm	Missed	Confusion
Aucun (baseline)	33,49 %	–	4,17 %	7,77 %	≈ 21,55 %
VAD	28,67 %	−4,82	0,03 %	6,87 %	21,77 %
Vote	31,63 %	−1,86	4,17 %	7,77 %	19,70 %
Nombre de locuteurs k	38,75 %	+5,26	4,17 %	7,77 %	26,82 %

1 La confusion (embeddings + clustering) est le composant qui limite la baseline : environ 21 à 22 points sur 33, présents dans tous les runs oracle.
2 La VAD coûte au maximum environ 5 points, surtout en false alarm.
3 Le vote coûte au maximum environ 2 points.
4 L’estimation de k n’est pas en cause : le vrai k dégrade le DER. ( sera en cause plus tard si le clustering est amelioré )

Idée : proposer une amelioration sur la VAD puis une amelioration sur le clustering 


## FICHIERS PROBLEMATIQUES 
certains fichier ont un DER plusieurs fois superieurs aux autre avec un facteur 10, meme avec le clustering en mode oracle :
-> ex de fichier problematique (mode oracle) : [73/90] tucrg | DER 474.2% | k 3 (réf 3) | RTF 0.002 (semble etre un concert)

tous environs les mmes points en communs : 
-> musique fortes ou bruit fort (bcp d'energie dans les frames, de bruit)
-> nbre de speakers important

la vad par energie est completement inefficace sur ces fichiers et le clustering a du mal, mme en connaissant le nombre de speaker.

## OBSERVATIONS : amelioration de la vad par claude code : 
temps de calcul multiplié par 3 : RTF 0.001 ou 2 avant -> 0.003 ou 5 mtn

┌────────────────────────────────────────────────┬───────────────────────┬────────┬────────┬─────────────────────┐
│                      Jeu                       │          VAD          │   FA   │  Miss  │ Erreur de détection │
├────────────────────────────────────────────────┼───────────────────────┼────────┼────────┼─────────────────────┤
│ Réglage (90 fichiers dev)                      │ energy_vad (actuelle) │ 4,18 % │ 2,47 % │ 6,65 %              │
├────────────────────────────────────────────────┼───────────────────────┼────────┼────────┼─────────────────────┤
│ Réglage (90 fichiers dev)                      │ gmm_vad (nouvelle)    │ 2,33 % │ 2,09 % │ 4,41 %              │
├────────────────────────────────────────────────┼───────────────────────┼────────┼────────┼─────────────────────┤
│ Test (90 fichiers, jamais utilisé pour régler) │ energy_vad (actuelle) │ 7,66 % │ 3,39 % │ 11,05 %             │
├────────────────────────────────────────────────┼───────────────────────┼────────┼────────┼─────────────────────┤
│ Test (90 fichiers, jamais utilisé pour régler) │ gmm_vad (nouvelle)    │ 4,10 % │ 2,52 % │ 6,62 %              │
└────────────────────────────────────────────────┴───────────────────────┴────────┴────────┴─────────────────────┘

- Ce qui a marché : gmm_vad utilise deux modèles statistiques, un pour la parole et un pour le reste, calculés à partir de caractéristiques du son (MFCC). Ils sont entraînés sur les 126 fichiers dev qui ne servent pas au réglage. Il n'y a aucun réseau pré-entraîné ni nouvelle dépendance (seulement sklearn, déjà installé). Sur le test, 74 fichiers s'améliorent et 16 se dégradent.
- Ce qui n'a pas marché : régler autrement la VAD par énergie plafonne vers 6,6 % sur le réglage. Le seuil du nouveau modèle change très peu le résultat.
- Erreurs restantes : ce sont surtout des fausses alarmes, concentrées sur quelques fichiers du test (wprog, wemos, kvkje, diysk). L'agent pense qu'il s'agit de musique ou de bruit de foule, mais il ne les a pas écoutés.
- Pas encore mesuré : l'effet sur le DER complet du pipeline de diarisation.


## LE CLUSTERING SPECTRAL, DES OBSERVATIONS PLEINES DE SURPRISES 

***OBSERVATION 1*** : sur le fichier problématique mentionné plus tot : tucrg | DER 474.2% | k 3 (réf 3) | RTF 0.002 (semble etre un concert)
le DER baisse considérablement avec la nouvelle vad mais L ESTIMATION DU NOMBRE DE SPEAKER QUI ETAIT CORRECTE DEVIENT FAUSSE : [73/90] tucrg | DER 48.1% | k 1 (réf 3) | RTF 0.007
hypothese : le nombre de locuteur était correcte par "chance" il devait se baser sur la musique et prendre celle-ci pour un locuteur
conclusion : ? 

*** OBSERVATION 2 *** : lorsque le nombre de locuteurs reel est connu par l'algorithme de clustering, le DER augmente ET u contraire Le JER s'améliore : de 61,13 à 53,46 %.

hypothese 1 : les embeddings etant bruité, on a des windows bruités eloignés des autres clusters, or le clustering spectral à tendance à vouloir former ces windows à former un cluster différent : force des clusters "artificiels" à se creer, on force des découpages en cluster que le clustering n'avait pas fais par lui meme. 

observations : le locuteur dominant est découpé : il occupe 2,96 clusters en moyenne, au lieu de 1,28 avec l'eigengap.

## À REVOIR : p_neighbors du clustering spectral (noté par Claude Code, 12/11)
- Fonctionnement : chaque window garde comme voisins ses p windows les plus similaires (cosinus), p = max(min_neighbors, round(p_neighbors × n)), n = nb de windows du fichier. Défaut p_neighbors = 0.1 (10 %).
- Problème : un locuteur qui a moins de p windows (moins de 10 % du fichier) est obligé de prendre des voisins chez les autres -> il est absorbé par le locuteur majoritaire. Probablement fréquent dans VoxConverse (présentateur dominant).
- Test synthétique (nuages gaussiens, PAS de vraies données) : groupes [300, 15] -> p = 0.1 : k = 1, ARI 0.0 ; p = 0.03 : k = 2, ARI 1.0. Groupes [400, 20, 20] -> p = 0.1 : k = 2, ARI 0.908 ; p = 0.03 : k = 3, ARI 0.955.
- Détail : une window se compte parmi ses propres voisins (cosinus avec elle-même = 1) -> p - 1 voisins réels.
***INTERESSANT***:
- Pistes : balayage de p_neighbors sur le dev (sample 90), ou choix automatique de p par fichier comme dans NME-SC (candidat d'amélioration pour l'ablation).
