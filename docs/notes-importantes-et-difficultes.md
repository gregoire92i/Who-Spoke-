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