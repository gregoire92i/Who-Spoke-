# Journal de recherche — VAD (Who Spoke, VoxConverse)

Protocole commun à toutes les itérations :
- Critère : detection error = (FA + MISS) / durée de parole de référence ; `DetectionErrorRate(collar=0.0, skip_overlap=False)`,
  UEM = [0, durée du wav], agrégation micro (identique à `main()` de `vad.py`).
- Référence : `energy_vad` d'`audio_features.py` (importée, non modifiée), avec les paramètres de `baseline.py` :
  trame 65 ms, hop 32,5 ms, seuil -30 dB, delta 3,18 dB et lissage 3 trames (défauts d'audio_features), min_silence 0,5 s, min_speech 0,25 s.
  Import : `from audio_features import energy_vad` (import simple). Historique : au départ, `rttm_files.py` (importé par
  `audio_features`) exécutait `save_sample_rttm(...)` au niveau module, et `vad.py` contournait cet effet de bord avec un module
  factice ; l'utilisateur a depuis commenté ces appels dans `rttm_files.py`, le contournement a été supprimé.
- Réglage rapide : `samples/dev/sample90_seed130.rttm` (90 fichiers dev, 7,405 h de parole sur 8,021 h ; 92,3 % de parole).
- Modèles appris : entraînés UNIQUEMENT sur les 126 fichiers dev hors sample90 (« dev-train »), réglés sur sample90.
- Validation finale (une seule fois, aucun réglage) : `samples/test/sample90_seed130.rttm`.
- Caractéristiques mises en cache (trame 65 ms / hop 32,5 ms, comme la baseline) : `python vad.py cache --split dev --rttm samples/dev/all_dev.rttm`.

Remarque de cadrage : la parole occupe 92,3 % du temps du sample de réglage. Une VAD « tout parole » aurait
FA = 7,7/92,3 = 8,3 % et Miss = 0 %. Les trous < 0,5 s de la référence représentent 209,9 s (0,79 % de la parole).

### Itération 0 — mesure de la référence
- Réflexion : point de départ, aucune modification.
- Hypothèse : aucune (mesure).
- Changement : aucun ; `energy_vad` (audio_features) avec seuil -30 dB, delta 3,18, lissage 3, min_silence 0,5, min_speech 0,25 (paramètres de baseline.py).
- Commande lancée : `python vad.py eval --method reference --per-file it0_reference_sample90.csv` (+ `--rttm samples/dev/all_dev.rttm`)
  Contrôle : `python vad.py eval --method energy` (ré-implémentation sur RMS en cache) donne exactement les mêmes chiffres.
- Résultats (sample90) : FA 4.177 % | Miss 2.472 % | critère 6.649 % (FA 1113.4 s, Miss 659.0 s, parole 26658.9 s)
  Résultats (all_dev, info) : FA 4.149 % | Miss 1.824 % | critère 5.973 % (FA 2824.2 s, Miss 1242.0 s, parole 68074.6 s)
- Analyse par fichier : erreurs concentrées. 10 fichiers = 48 % du FA et 70 % du Miss.
  Pires : kdfqk (FA 165.0 s, Miss 12.2 s), hgdez (FA 17.6, Miss 125.3), hkzpa (FA 15.5, Miss 77.4), kkghn (FA 31.3, Miss 51.1),
  iwdjy (FA 73.1, Miss 2.4), xypdm (FA 41.6, Miss 14.0), evtyi (FA 10.0, Miss 45.5), zrlyl (Miss 50.6), oxxwk (FA 51.1), bauzd (Miss 52.3).
  Deux familles : fichiers où le non-parole est énergétique (musique/bruit, FA) et fichiers où de la parole est > 30 dB sous le max (Miss).
- Interprétation : —
- Décision : référence = 6.649 %.

### Itération 1 — énergie normalisée par le niveau médian du fichier
- Réflexion : les Miss de hgdez/hkzpa/bauzd/zrlyl viennent de parole à plus de 30 dB sous le max du fichier : un seul pic fort
  (applaudissements, jingle) fixe la référence 0 dB. Le max est une statistique fragile.
- Hypothèse : si je normalise l'énergie par le percentile 50 du fichier (au lieu du max), alors le Miss doit baisser sur ces fichiers
  sans hausse équivalente du FA, parce que la médiane (92 % de parole) suit le niveau typique de la parole.
- Changement : `vad_energy(level_ref="pct", pct=50)`, balayage seuil_db ∈ {-20, -18, -16, -14, -12, -10, -8} (autres paramètres = référence).
- Commande lancée : `python vad.py eval --method energy --set level_ref=pct pct=50 --sweep seuil_db=-20,-18,-16,-14,-12,-10,-8`
- Résultats :
  -20 : FA 6.182 % | Miss 0.771 % | critère 6.953 %
  -18 : FA 5.655 % | Miss 1.064 % | critère 6.718 %
  -16 : FA 5.189 % | Miss 1.417 % | critère 6.606 %   <- meilleur
  -14 : FA 4.773 % | Miss 1.879 % | critère 6.652 %
  -12 : FA 4.450 % | Miss 2.413 % | critère 6.863 %
  -10 : FA 3.851 % | Miss 3.100 % | critère 6.951 %
   -8 : FA 3.319 % | Miss 4.092 % | critère 7.411 %
  (référence : FA 4.177 | Miss 2.472 | 6.649 ; meilleure précédente : idem)
- Interprétation : non concluante. Le Miss baisse bien (2.472 -> 1.417 % à -16 dB) mais le FA monte d'autant (4.177 -> 5.189 %) :
  gain net 0.043 pt, au niveau du bruit. Observé : quel que soit le réglage, FA+Miss reste entre 6.6 et 7.4 %.
  Supposé : l'énergie ne sépare pas parole et non-parole énergétique (musique, bruit, rires) ; changer la normalisation déplace
  seulement le compromis FA/Miss.
- Remise en question : je considère l'approche « énergie seule » arrivée au maximum de son potentiel (~6.6 %) : les deux balayages
  (normalisation max ou médiane) donnent la même courbe FA/Miss. Il faut une information spectrale qui distingue la parole de la
  musique/du bruit -> changement d'approche : modèle appris.
- Décision : abandonner (meilleure version reste la référence).

### Itération 2 — nouvelle approche : GMM parole / non-parole sur MFCC
- Réflexion : les erreurs dominantes (kdfqk FA 165 s, iwdjy FA 73 s, oxxwk FA 51 s ; hgdez Miss 125 s) sont des cas où l'énergie
  du non-parole est comparable à celle de la parole. L'enveloppe spectrale (MFCC) de la parole est différente de celle de la musique
  ou du bruit, indépendamment du niveau.
- Hypothèse : si je remplace le seuil d'énergie par le log-rapport de vraisemblance de deux GMM (parole / non-parole) appris sur
  des MFCC normalisés par fichier (CMVN), alors FA ET Miss doivent baisser sur les fichiers à musique/bruit, parce que la décision
  repose sur la forme spectrale et plus sur le niveau absolu.
- Changement : `vad_model(kind="gmm", feats="mfcc")` : 13 MFCC (depuis le log-mel 40 bandes de la même STFT 65/32.5 ms) + deltas
  (largeur 5), CMVN par fichier ; 2 GMM diagonales 16 composantes (sklearn GaussianMixture, random_state=0), entraînées sur les
  126 fichiers dev hors sample90, 1 trame sur 4. Décision LLR > seuil, puis même post-traitement (min_silence 0.5, min_speech 0.25).
  Balayage seuil (LLR) ∈ {-4, -2, -1, -0.5, 0, 0.25, 0.5, 1, 2}.
- Commandes lancées : `python vad.py eval --method model --set kind=gmm feats=mfcc --sweep seuil=-4,-2,-1,0,1,2`
  puis `--sweep seuil=-0.5,0.25,0.5`, puis `--set kind=gmm feats=mfcc seuil=0 --per-file it2_gmm_sample90.csv`
- Résultats :
  -4   : FA 6.763 % | Miss 0.119 % | critère 6.882 %
  -2   : FA 5.717 % | Miss 0.314 % | critère 6.030 %
  -1   : FA 4.473 % | Miss 0.659 % | critère 5.133 %
  -0.5 : FA 3.717 % | Miss 1.094 % | critère 4.811 %
   0   : FA 2.783 % | Miss 1.698 % | critère 4.481 %
   0.25: FA 2.326 % | Miss 2.085 % | critère 4.411 %   <- meilleur
   0.5 : FA 1.961 % | Miss 2.485 % | critère 4.446 %
   1   : FA 1.435 % | Miss 3.349 % | critère 4.784 %
   2   : FA 0.896 % | Miss 5.312 % | critère 6.207 %
  (référence : FA 4.177 | Miss 2.472 | 6.649 ; meilleure précédente : référence 6.649)
- Analyse par fichier (seuil 0 vs référence) : 64 fichiers améliorés, 25 dégradés. Gros gains : kdfqk -120.2 s, hgdez -82.7 s,
  hkzpa -58.2 s, kkghn -41.5 s, iwdjy -40.3 s, bauzd -38.8 s. Dégradations : grzbb +63.8 s (FA 21.2 -> 91.6 s), evtyi +13.7 s
  (FA 10.0 -> 53.9 s), qhesr +7.1 s (Miss 0.2 -> 11.3 s). Pires fichiers restants : grzbb, evtyi, hgdez, xypdm, kdfqk.
- Interprétation : confirmée. FA (4.177 -> 2.326 %) et Miss (2.472 -> 2.085 %) baissent tous les deux, critère 6.649 -> 4.411 %
  (-2.238 pt). Entre seuil 0, 0.25 et 0.5 l'écart est <= 0.07 pt : plateau, le choix 0.25 dans ce plateau n'est pas significatif.
  Supposé (non vérifié à l'écoute) : grzbb et evtyi contiennent du non-parole dont l'enveloppe ressemble à de la parole
  (voix chantée, foule) que la GMM accepte.
- Décision : garder (seuil 0.25). Critère < 5 % atteint -> arrêt de la boucle, validation sur le test.

### Validation finale (test, samples/test/sample90_seed130.rttm) — exécutée une seule fois, aucun réglage
- Commandes : `python vad.py cache --split test --rttm samples/test/sample90_seed130.rttm`, puis
  `python vad.py eval --method reference --split test --rttm samples/test/sample90_seed130.rttm --per-file val_reference_test90.csv`
  `python vad.py eval --method model --split test --rttm samples/test/sample90_seed130.rttm --set kind=gmm feats=mfcc seuil=0.25 --per-file val_gmm_test90.csv`
- Référence energy_vad : FA 7.658 % | Miss 3.387 % | critère 11.045 % (FA 4401.4 s, Miss 1946.7 s, parole 57473.3 s)
- GMM MFCC seuil 0.25 : FA 4.100 % | Miss 2.522 % | critère 6.622 % (FA 2356.2 s, Miss 1449.7 s, parole 57473.3 s)
- 74 fichiers améliorés, 16 dégradés. L'objectif < 5 % atteint sur le réglage (4.411 %) N'EST PAS atteint sur la validation (6.622 %).
  Le test est plus difficile pour les deux VAD (référence 11.045 % contre 6.649 % sur le réglage). Pires fichiers test (GMM) :
  wprog (FA 115.9 s, Miss 101.2 s), wemos (FA 122.5), kvkje (FA 114.3), diysk (FA 95.1), bgvvt (FA 72.1, Miss 31.8).

---
## Reprise de la boucle (critère d'arrêt abaissé à 3 % par l'utilisateur ; définition d'agent relue : seul le critère a changé,
## plus une idée ajoutée : « clustering poubelle » bruit/musique). Point de départ : gmm_vad, 4.411 %. Le test a déjà été regardé
## une fois (validation de gmm_vad) ; il ne sert toujours qu'à la validation finale.

### Itération 3 — classifieur HGB sur caractéristiques riches avec contexte
- Réflexion : avec la GMM, les erreurs restantes sont surtout des FA sur des fichiers à musique/foule (evtyi FA 45.1 s, xypdm 36.8 s,
  hgdez 30.7 s, iwdjy 27.2 s) et des Miss dispersés (kkghn 35.9 s, hgdez 35.9 s, kdfqk 34.5 s). La GMM ne voit qu'une trame de
  65 ms sans contexte ni information d'harmonicité. La musique est périodique mais spectralement stable, et la foule est apériodique.
  La parole alterne voisé et non voisé à ~4 Hz.
- Remise en question : la GMM diagonale sur 26 dimensions trame par trame est limitée (pas de contexte, frontières de décision
  simples). Je ne pense pas qu'un réglage de plus (nombre de composantes) suffise pour passer de 4.4 à 3 %.
- Hypothèse : si je remplace la GMM MFCC par un classifieur discriminant (HistGradientBoosting) sur log-mel normalisé + niveaux
  relatifs (au p95 et au p5 du fichier) + platitude, entropie, centroïde, ZCR, périodicité + statistiques de contexte (moyenne et
  écart-type sur 11 et 31 trames, variabilité spectrale long terme), alors le FA sur musique/foule et le Miss doivent baisser,
  parce que le contexte et l'harmonicité séparent ces classes mieux que l'enveloppe instantanée.
- Changement : `vad_model(kind="hgb", feats="base")` (`build_X` base, 137 dimensions), HGB 300 itérations, 63 feuilles, lr 0.1,
  random_state 0, entraîné sur dev-train (126 fichiers, 1 trame sur 4). Post-traitement inchangé. Balayage seuil (log-odds) ∈
  {-3, -2, -1, -0.5, 0, 0.5, 0.75, 1, 1.25, 1.5, 2}.
- Commandes : `python vad.py eval --method model --set kind=hgb feats=base --sweep seuil=-3,-2,-1,-0.5,0,0.5,1,2` ;
  `--sweep seuil=0.75,1.25,1.5` ; `--set kind=hgb feats=base seuil=1 --per-file it3_hgb_sample90.csv`
- Résultats :
  -3   : FA 6.443 % | Miss 0.005 % | critère 6.448 %
  -2   : FA 5.553 % | Miss 0.035 % | critère 5.589 %
  -1   : FA 4.461 % | Miss 0.169 % | critère 4.630 %
  -0.5 : FA 3.896 % | Miss 0.316 % | critère 4.212 %
   0   : FA 3.302 % | Miss 0.557 % | critère 3.859 %
   0.5 : FA 2.749 % | Miss 0.911 % | critère 3.660 %
   0.75: FA 2.466 % | Miss 1.127 % | critère 3.594 %
   1   : FA 2.183 % | Miss 1.349 % | critère 3.531 %   <- meilleur
   1.25: FA 1.972 % | Miss 1.648 % | critère 3.619 %
   1.5 : FA 1.757 % | Miss 2.025 % | critère 3.781 %
   2   : FA 1.392 % | Miss 2.789 % | critère 4.181 %
  (référence : FA 4.177 | Miss 2.472 | 6.649 ; meilleure précédente gmm_vad : FA 2.326 | Miss 2.085 | 4.411)
- Analyse par fichier (vs gmm_vad seuil 0.25) : 63 améliorés, 26 dégradés. Gains : hgdez -23.2 s, xypdm -20.8 s, evtyi -10.2 s.
  Dégradations faibles : lknjp +8.2 s (FA), willh +5.1, pnook +5.0. Pires restants : kdfqk (FA 40.1, Miss 19.0), evtyi (FA 43.7,
  Miss 9.1), hgdez (22.0 / 21.4), kkghn (Miss 34.0), xypdm (17.3 / 18.7).
- Interprétation : confirmée. FA (2.326 -> 2.183 %) et Miss (2.085 -> 1.349 %) baissent ; critère -0.880 pt vs gmm_vad,
  -3.118 pt vs référence. Je n'ai pas fait d'ablation, donc je ne sais pas quelle part du gain vient du contexte, de la
  périodicité ou du classifieur lui-même.
- Décision : garder (hgb, seuil 1). 3.531 % > 3 % : on continue.

### Itération 4 — min_silence plus court (post-traitement)
- Réflexion : la référence contient 209.9 s de trous < 0.5 s (0.79 % de la parole) ; min_silence = 0.5 s les comble tous -> FA.
- Hypothèse : si je réduis min_silence, alors le FA doit baisser (pauses courtes rendues) sans hausse équivalente du Miss.
- Changement : `vad_model(kind=hgb, feats=base, seuil=1)`, balayage min_silence ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.7}.
- Commande : `python vad.py eval --method model --set kind=hgb feats=base seuil=1 --sweep min_silence=0.1,0.2,0.3,0.4,0.5,0.7`
- Résultats :
  0.1 : FA 1.697 % | Miss 2.395 % | critère 4.093 %
  0.2 : FA 1.823 % | Miss 2.069 % | critère 3.891 %
  0.3 : FA 1.907 % | Miss 1.819 % | critère 3.726 %
  0.4 : FA 2.009 % | Miss 1.611 % | critère 3.620 %
  0.5 : FA 2.183 % | Miss 1.349 % | critère 3.531 %   <- (= meilleure actuelle)
  0.7 : FA 2.704 % | Miss 0.862 % | critère 3.566 %
  (référence : 6.649 ; meilleure précédente : 3.531)
- Interprétation : réfutée. Le FA baisse (2.183 -> 1.697 %) mais le Miss monte plus (1.349 -> 2.395 %) : les trous détectés par
  le classifieur sont en majorité de faux trous dans la parole, que le comblement à 0.5 s corrige.
- Décision : abandonner (min_silence reste 0.5).

### Itération 5 — lissage des log-odds avant seuil
- Réflexion : l'itération 4 montre des décisions trame par trame bruitées (faux trous courts).
- Hypothèse : si je lisse les log-odds par moyenne glissante avant le seuil, alors les faux trous et les faux îlots diminuent
  -> Miss et FA baissent.
- Changement : `smooth_frames` ∈ {1, 3, 5, 9, 15, 25} (seuil 1, min_silence 0.5).
- Commande : `python vad.py eval --method model --set kind=hgb feats=base seuil=1 --sweep smooth_frames=1,3,5,9,15,25`
- Résultats :
  1 : FA 2.183 % | Miss 1.349 % | critère 3.531 %
  3 : FA 2.011 % | Miss 1.557 % | critère 3.569 %
  5 : FA 2.000 % | Miss 1.571 % | critère 3.571 %
  9 : FA 2.046 % | Miss 1.475 % | critère 3.521 %
  15: FA 2.205 % | Miss 1.338 % | critère 3.543 %
  25: FA 2.506 % | Miss 1.136 % | critère 3.643 %
- Interprétation : non concluante. Meilleur 3.521 % (9 trames) contre 3.531 % : 0.010 pt, c'est du bruit. Le comblement
  min_silence/min_speech fait déjà ce travail.
- Décision : abandonner (smooth_frames = 1).

### Diagnostic (entre itérations 5 et 6) — nature des erreurs restantes (hgb, seuil 1), grille 10 ms
- FA 584 s : 411.8 s à moins de 0.25 s d'une frontière de la référence, 172.6 s plus loin. Miss : 165.0 s près d'une frontière,
  195.3 s plus loin.
- Longueur des morceaux d'erreur : FA < 0.5 s : 1911 morceaux, 296.0 s ; 0.5-2 s : 278, 210.1 s ; 2-10 s : 17, 78.2 s ; > 10 s : 0.
  Miss < 0.5 s : 1186, 174.3 s ; 0.5-2 s : 240, 169.5 s ; 2-10 s : 5, 16.5 s ; > 10 s : 0.
- Plancher oracle (vérité terrain passée dans le post-traitement) : min_silence 0.5 -> FA 0.865 % | Miss 0.072 % | 0.936 % ;
  min_silence 0.3 -> 0.538 % ; 0.2 -> 0.416 % ; 0.1 -> 0.307 % ; 0 -> FA 0.131 % | Miss 0.130 % | 0.262 % (quantification 32.5 ms).
- Conclusion : les grosses régions de musique/bruit ne sont plus le problème principal. L'erreur restante est faite de
  frontières imprécises et de pauses courtes. Le post-traitement à 0.5 s coûte à lui seul 0.94 pt.

### Itération 6 — min_silence réduit avec seuil ré-ajusté
- Réflexion : à l'itération 4, le seuil (1) avait été réglé pour min_silence = 0.5 ; avec moins de comblement, un seuil plus
  bas (moins de faux trous) pourrait être optimal.
- Hypothèse : si je prends min_silence 0.2 ou 0.3 ET que je ré-balaye le seuil, alors le critère passe sous 3.531 %, parce que le
  plancher oracle baisse de 0.936 à 0.416/0.538 %.
- Changement : balayage seuil ∈ {-1, -0.5, 0, 0.5, 1} pour min_silence ∈ {0.2, 0.3}.
- Commande : `for ms in 0.2 0.3; do python vad.py eval --method model --set kind=hgb feats=base min_silence=$ms --sweep seuil=-1,-0.5,0,0.5,1; done`
- Résultats :
  ms 0.2 : -1 : 3.950/0.344/4.294 ; -0.5 : 3.391/0.571/3.962 ; 0 : 2.819/0.899/3.718 ; 0.5 : FA 2.282 % | Miss 1.396 % | 3.678 % ; 1 : 1.823/2.069/3.891
  ms 0.3 : -1 : 4.100/0.286/4.385 ; -0.5 : 3.529/0.477/4.007 ; 0 : 2.955/0.785/3.740 ; 0.5 : FA 2.397 % | Miss 1.218 % | 3.615 % ; 1 : 1.907/1.819/3.726
  (format FA/Miss/critère en % ; meilleure précédente : 3.531)
- Interprétation : réfutée. Le meilleur point (3.615 %) reste moins bon que 3.531 %. Le classifieur ne localise pas assez bien
  les pauses courtes pour profiter d'un comblement plus court. Supposé : les statistiques de contexte (11 et 31 trames)
  étalent les creux d'énergie.
- Remise en question : le post-traitement est à bout (itérations 4, 5, 6). Le gain doit venir de la décision trame par trame
  (classifieur / caractéristiques).
- Décision : abandonner.

### Itération 7 — caractéristiques locales sensibles aux pauses (feats="base+local")
- Réflexion : diagnostic -> l'erreur est dominée par les frontières et les pauses courtes ; les statistiques de contexte
  (11/31 trames) noient les creux brefs.
- Hypothèse : si j'ajoute des niveaux relatifs AU VOISINAGE PROCHE (énergie totale et énergie 300-3400 Hz moins leur max sur 9 et
  31 trames, moins leur moyenne sur 11 et 31 trames, min sur 5 trames moins moyenne 31, pente, périodicité moins sa moyenne 31),
  alors FA et Miss près des frontières doivent baisser, parce qu'un creux de 0.2 s devient visible indépendamment du niveau
  absolu.
- Changement : `build_X(feats="base+local")` (+13 dimensions), HGB inchangé, balayage seuil ∈ {0.5, 0.75, 1, 1.25, 1.5}.
- Commande : `python vad.py eval --method model --set kind=hgb feats=base+local --sweep seuil=0.5,0.75,1,1.25,1.5`
- Résultats :
  0.5 : FA 2.581 % | Miss 0.872 % | critère 3.453 %
  0.75: FA 2.335 % | Miss 1.076 % | critère 3.410 %
  1   : FA 2.090 % | Miss 1.302 % | critère 3.392 %   <- meilleur
  1.25: FA 1.878 % | Miss 1.564 % | critère 3.442 %
  1.5 : FA 1.700 % | Miss 1.867 % | critère 3.567 %
  (référence : 6.649 ; meilleure précédente : 3.531 (FA 2.183, Miss 1.349))
- Interprétation : confirmée, gain faible. -0.139 pt ; FA -0.093 pt, Miss -0.047 pt. Le gain se retrouve à chaque seuil
  (0.5 : 3.660 -> 3.453 ; 1.5 : 3.781 -> 3.567), ce qui le rend plus crédible qu'un point isolé. Mais il reste du même ordre que
  la variabilité entre modèles observée à l'itération 9 (~0.05 pt).
- Décision : garder (base+local, seuil 1).

### Itération 8 — contexte long (2 s et 4 s) (feats="base+local+long")
- Réflexion : il reste 172.6 s de FA et 195.3 s de Miss loin des frontières (musique, foule, parole sur fond sonore).
- Hypothèse : si j'ajoute moyenne et écart-type sur 61 et 121 trames (énergie, périodicité, entropie, platitude, variabilité
  spectrale), alors le FA sur la musique/foule doit baisser, parce que leur stationnarité long terme diffère de la parole.
- Changement : `feats="base+local+long"`, balayage seuil ∈ {0.5, 0.75, 1, 1.25, 1.5}.
- Commande : `python vad.py eval --method model --set kind=hgb feats=base+local+long --sweep seuil=0.5,0.75,1,1.25,1.5`
- Résultats : 0.5 : 2.568/0.926/3.494 ; 0.75 : FA 2.334 % | Miss 1.114 % | 3.448 % ; 1 : 2.139/1.340/3.479 ; 1.25 : 1.939/1.580/3.519 ;
  1.5 : 1.753/1.867/3.620  (FA/Miss/critère en % ; meilleure précédente : 3.392)
- Interprétation : réfutée. C'est un peu moins bon à chaque seuil (+0.04 à +0.09 pt). Supposé : avec seulement 126 fichiers
  d'entraînement, le contexte long apprend des particularités de fichiers plutôt que des propriétés générales.
- Décision : abandonner.

### Itération 9 — 4x plus de trames d'entraînement (step=1)
- Hypothèse : si j'entraîne sur toutes les trames de dev-train (step 1 au lieu de 4), alors le classifieur généralise mieux et
  le critère baisse.
- Changement : `step=1` (feats base+local), balayage seuil ∈ {0.5, 0.75, 1, 1.25, 1.5}.
- Commande : `python vad.py eval --method model --set kind=hgb feats=base+local step=1 --sweep seuil=0.5,0.75,1,1.25,1.5`
- Résultats : 0.5 : 2.478/1.000/3.478 ; 0.75 : FA 2.192 % | Miss 1.254 % | 3.446 % ; 1 : 1.937/1.514/3.452 ; 1.25 : 1.727/1.827/3.554 ;
  1.5 : 1.552/2.211/3.764  (meilleure précédente : 3.392)
- Interprétation : réfutée. Pas mieux (+0.054 pt au meilleur point). Les trames consécutives sont très corrélées : sous-échantillonner
  à 1/4 ne perd presque pas d'information. Cet écart donne une idée du bruit d'un modèle à l'autre (~0.05 pt).
- Décision : abandonner (step 4).

### Itération 10 — décalage des frontières des segments (d_start, d_end)
- Réflexion : 411.8 s de FA à moins de 0.25 s d'une frontière. J'ai supposé des segments trop longs (contexte qui déborde).
- Hypothèse : si je rétrécis les segments (d_start > 0 ou d_end > 0), alors le FA baisse plus que le Miss ne monte.
- Changement : `adjust_bounds` dans `vad_model`. Balayage d_start ∈ {-0.06, -0.03, 0.03, 0.06, 0.1} (d_end = 0), puis
  d_end ∈ {-0.3, -0.2, -0.15, -0.1, -0.06, -0.03, 0.03, 0.06, 0.1} (d_start = 0). Valeur négative = allonger.
- Commandes : `python vad.py eval --method model --set kind=hgb feats=base+local seuil=1 --sweep d_start=-0.06,-0.03,0.03,0.06,0.1`
  puis `--sweep d_end=-0.06,-0.03,0.03,0.06,0.1` puis `--sweep d_end=-0.1,-0.15,-0.2,-0.3`
- Résultats :
  d_start : -0.06 : 2.310/1.186/3.496 ; -0.03 : 2.190/1.234/3.424 ; 0.03 : 2.008/1.388/3.395 ; 0.06 : 1.941/1.489/3.430 ; 0.1 : 1.871/1.644/3.515
  d_end : -0.3 : 3.005/0.558/3.563 ; -0.2 : 2.634/0.738/3.372 ; -0.15 : 2.470/0.849/3.319 ;
          -0.1 : FA 2.320 % | Miss 0.976 % | critère 3.296 %   <- meilleur ;
          -0.06 : 2.214/1.092/3.306 ; -0.03 : 2.147/1.192/3.339 ; 0.03 : 2.044/1.424/3.468 ; 0.06 : 2.009/1.557/3.565 ; 0.1 : 1.973/1.745/3.718
  (meilleure précédente : 3.392 = FA 2.090, Miss 1.302)
- Interprétation : hypothèse réfutée (rétrécir n'aide jamais), mais l'observation inverse est utile. Allonger les FINS de segments
  de 0.06 à 0.15 s réduit le critère : -0.096 pt à d_end = -0.1 (Miss 1.302 -> 0.976 %, FA 2.090 -> 2.320 %). Les débuts sont bien
  placés (optimum plat autour de 0). Supposé : le classifieur coupe trop tôt les fins de mots (décroissance d'énergie, fricatives
  finales), alors que les annotateurs les incluent. Même modèle, post-traitement seul : ce n'est pas de la variance de modèle,
  mais le gain reste petit.
- Décision : garder d_end = -0.1 (plateau de -0.06 à -0.15). Meilleure actuelle : 3.296 %.
- Remise en question : l'approche « classifieur trame par trame sur 126 fichiers » plafonne vers 3.3-3.4 %. Les itérations 5, 8
  et 9 sont neutres ou négatives, et les gains des itérations 7 et 10 sont proches du bruit. Je change d'axe : adaptation au
  fichier (idée « clustering poubelle »), car les erreurs restantes dépendent beaucoup du fichier (fond sonore, micro).

### Itération 11 — adaptation au fichier (« modèle poubelle » non supervisé)
- Réflexion : erreurs restantes très dépendantes du fichier (kdfqk 60.9 s, evtyi 41.1 s, hgdez 39.3 s, xypdm 38.1 s ; les 5 pires
  fichiers = 0.8 pt). Le fond sonore propre à chaque fichier n'est pas vu à l'entraînement.
- Hypothèse : si j'ajoute au log-odds global alpha × LLR de deux GMM (4 composantes diagonales, MFCC CMVN + deltas) apprises SUR LE
  FICHIER à partir des trames où le classifieur global est sûr (|log-odds| > 3), alors le FA sur le bruit/musique propre au
  fichier doit baisser, parce que ces GMM modélisent précisément ce fond.
- Changement : `file_adapted_llr` + `vad_model(adapt=alpha)`. Balayage alpha ∈ {0, 0.05, 0.1, 0.2, 0.4} (seuil 1), puis seuil
  ∈ {1.25, 1.5, 2, 2.5} pour alpha 0.2 et {1.25, 1.5} pour alpha 0.1 (base+local, d_end -0.1).
- Commandes : `python vad.py eval --method model --set kind=hgb feats=base+local seuil=1 d_end=-0.1 --sweep adapt=0,0.05,0.1,0.2,0.4`
  `python vad.py eval --method model --set kind=hgb feats=base+local d_end=-0.1 adapt=0.2 --sweep seuil=1.25,1.5,2,2.5` (et adapt=0.1 seuil=1.25,1.5)
- Résultats :
  alpha 0 : 2.320/0.976/3.296 ; 0.05 : FA 2.300 % | Miss 0.938 % | 3.238 % ; 0.1 : 2.403/0.868/3.271 ; 0.2 : 2.541/0.782/3.323 ; 0.4 : 2.700/0.742/3.442
  alpha 0.2 : seuil 1.25 : 2.423/0.888/3.312 ; 1.5 : 2.313/0.987/3.299 ; 2 : 2.149/1.195/3.344 ; 2.5 : 1.957/1.480/3.437
  alpha 0.1 : seuil 1.25 : 2.253/0.993/3.246 ; 1.5 : 2.107/1.137/3.244
  (meilleure précédente : 3.296)
- Interprétation : non concluante. Gain maximal 0.058 pt (alpha 0.05), au niveau du bruit. Plus alpha est grand, plus le FA monte :
  le terme adaptatif ajoute surtout un biais vers la parole, plutôt qu'une séparation propre au fichier. Observé (itération 3
  et suivantes) : les erreurs restantes sont surtout des frontières, que ce modèle trame par trame ne corrige pas.
- Décision : abandonner (complexité non justifiée).

### Itération 12 — empilement temporel (stacking) à deux étages
- Réflexion : morceaux d'erreur fragmentés (kdfqk : 60.9 s d'erreur dont 9.3 s seulement en morceaux > 0.8 s). Les erreurs
  sont des frontières et des pauses mal localisées. Le seul indice temporel exploité aujourd'hui est le comblement à 0.5 s.
- Hypothèse : si un 2e classifieur reçoit le contexte temporel des log-odds du 1er (décalages ±1 à ±20 trames,
  min/max/moyenne sur 5, 11, 31 trames) + les caractéristiques du 1er étage, alors il apprend les motifs de pause et de fin
  de mot -> FA et Miss baissent près des frontières.
- Changement : `vad_stack` : 1er étage = HGB base+local (inchangé). Les log-odds d'entraînement du 2e étage sont calculés
  HORS-ÉCHANTILLON (validation croisée 3 plis par fichier sur dev-train). 2e étage HGB (300 it., 63 feuilles), d_end -0.1.
  Balayage seuil ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2}, puis d_end ∈ {0, -0.05, -0.1, -0.15} à seuil 0.5.
- Commandes : `python vad.py eval --method stack --set d_end=-0.1 --sweep seuil=0,0.5,1,1.5,2` ;
  `--set seuil=0.5 --sweep d_end=0,-0.05,-0.1,-0.15` ; `--set d_end=-0.1 --sweep seuil=0.25,0.75`
- Résultats :
  seuil 0 : 2.453/0.729/3.182 ; 0.25 : 2.206/0.921/3.127 ; 0.5 : FA 2.003 % | Miss 1.129 % | critère 3.131 % ; 0.75 : 1.802/1.369/3.171 ;
  1 : 1.646/1.612/3.258 ; 1.5 : 1.336/2.345/3.681 ; 2 : 1.061/3.369/4.430
  d_end (seuil 0.5) : 0 : 1.756/1.393/3.149 ; -0.05 : 1.869/1.250/3.119 ; -0.1 : 2.003/1.129/3.131 ; -0.15 : 2.152/1.024/3.176
  (référence 6.649 ; meilleure précédente 3.296 = FA 2.320, Miss 0.976)
- Interprétation : confirmée, gain modéré. -0.165 pt (3.296 -> 3.131) au réglage gardé. FA -0.317 pt, Miss +0.153 pt (le
  compromis se déplace aussi). Le gain se retrouve sur tout le plateau (3.12-3.18 entre seuil 0 et 0.75), donc ce n'est pas un
  point isolé. Le rôle de d_end s'est presque effacé (plat entre 0 et -0.1) : le 2e étage a appris une partie du biais des fins.
- Décision : garder (stack, seuil 0.5, d_end -0.1). Meilleure : 3.131 %.

### Itération 13 — min_silence plus court avec le 2e étage
- Hypothèse : le 2e étage localise mieux les pauses ; si je réduis min_silence (0.2, 0.3, 0.4) en rebalayant le seuil, alors le FA
  dû aux pauses comblées (plancher oracle 0.865 %) doit baisser.
- Commande : `for ms in 0.2 0.3 0.4; do python vad.py eval --method stack --set d_end=-0.1 min_silence=$ms --sweep seuil=-0.5,0,0.5,1; done`
- Résultats (FA/Miss/critère %) :
  ms 0.2 : -0.5 : 2.796/0.568/3.364 ; 0 : 2.274/0.889/3.163 ; 0.5 : 1.842/1.354/3.196 ; 1 : 1.468/2.025/3.493
  ms 0.3 : -0.5 : 2.849/0.539/3.388 ; 0 : 2.322/0.840/3.161 ; 0.5 : 1.892/1.272/3.164 ; 1 : 1.525/1.912/3.437
  ms 0.4 : -0.5 : 2.908/0.504/3.412 ; 0 : 2.375/0.790/3.165 ; 0.5 : 1.934/1.203/3.136 ; 1 : 1.582/1.756/3.338
  (meilleure précédente : 3.131)
- Interprétation : réfutée. Aucun point ne bat 3.131 %. Chaque pause réelle rendue est compensée par une coupure parasite.
- Décision : abandonner.

### Diagnostic (avant itération 14) — sensibilité aux frontières (collar diagnostique, PAS le critère)
- `stack` seuil 0.5 d_end -0.1, DetectionErrorRate avec collar : 0 -> 3.131 % ; ±0.05 s -> 2.430 % (FA 1.502, Miss 0.927) ;
  ±0.10 s -> 1.951 % ; ±0.20 s -> 1.349 %.
- La bande ±50 ms autour des frontières de référence pèse 0.70 pt. La quantification à 32.5 ms n'en explique qu'environ
  0.13 pt (oracle 0.262 % sans comblement). Le reste vient du placement imprécis des frontières par le classifieur.

### Itération 14 — affinage heuristique des frontières par énergie fine (25 ms / 10 ms)
- Hypothèse : si je déplace chaque frontière (dans ±W = 0.1 s) au point où l'énergie 300-3400 Hz fine (fenêtre 25 ms, hop 10 ms)
  franchit plancher + ratio × (niveau parole − plancher) (p10/p90 locaux sur 0.5 s), alors l'erreur dans la bande ±50 ms
  doit baisser, parce que l'énergie fine localise les attaques et les extinctions à 10 ms près.
- Changement : `vad_stack_refined` (= vad_stack seuil 0.5, d_end 0, + `refine_bounds`), cache `results/vad_agent/cache_fine/`.
  Balayage ratio ∈ {0.1, 0.2, 0.3, 0.5} (W 0.1). L'affinage remplace d_end.
- Commandes : `python -c "import vad; vad.build_fine_cache('dev','samples/dev/sample90_seed130.rttm',3)"` ;
  `python vad.py eval --method stack_refined --set seuil=0.5 W=0.1 --sweep ratio=0.1,0.2,0.3,0.5`
- Résultats : 0.1 : FA 2.027 % | Miss 1.082 % | critère 3.109 % ; 0.2 : 1.897/1.217/3.114 ; 0.3 : 1.843/1.310/3.153 ; 0.5 : 1.800/1.358/3.159
  (sans affinage ni d_end : 1.756/1.393/3.149 ; meilleure précédente : 3.131)
- Interprétation : non concluante. -0.022 pt contre la meilleure, -0.040 pt contre le même étage sans d_end : bruit. Supposé : le
  seuil d'énergie local est piégé par la parole voisine, le souffle et les fricatives ; une règle d'énergie seule ne sait pas où
  l'annotateur place la frontière.
- Décision : abandonner. Suite : décision apprise directement à 10 ms (2e étage sur grille fine).

### Itération 15 — 2e étage appris sur grille fine 10 ms (`vad_fine`)
- Réflexion : la règle d'énergie (itération 14) ne sait pas où l'annotateur met la frontière ; un modèle appris à 10 ms le pourrait.
- Hypothèse : si le 2e étage décide toutes les 10 ms (log-odds hors-échantillon du 1er étage interpolés, contexte ±0.6 s ; énergie
  fine 300-3400 Hz relative au p95 du fichier et au max local, contexte ±0.3 s ; min/max glissants ; X du 1er étage), alors
  l'erreur dans la bande ±50 ms des frontières baisse et le critère passe sous 3.131 %.
- Changement : `vad_fine` (remplace `vad_stack`), HGB 300 it., 63 feuilles, dev-train 1 trame fine sur 12. Balayage seuil ∈
  {0, 0.25, 0.5, 0.75, 1} (d_end 0), puis d_end ∈ {-0.05, -0.1} à seuil 0.25.
- Commandes : `python -c "import vad; vad.build_fine_cache('dev','samples/dev/all_dev.rttm',3)"` ;
  `python vad.py eval --method fine --sweep seuil=0,0.25,0.5,0.75,1` ; `python vad.py eval --method fine --set seuil=0.25 --sweep d_end=-0.05,-0.1`
- Résultats :
  seuil 0 : 2.366/0.736/3.102 ; 0.25 : FA 2.131 % | Miss 0.951 % | critère 3.082 % ; 0.5 : 1.906/1.177/3.083 ; 0.75 : 1.737/1.420/3.157 ; 1 : 1.569/1.682/3.251
  d_end -0.05 : FA 2.240 % | Miss 0.831 % | critère 3.070 % ; d_end -0.1 : 2.369/0.732/3.101
  Diagnostic collar (seuil 0.25, d_end 0) : ±0.05 s -> 2.405 % ; ±0.10 s -> 1.947 % (stack : 2.430 / 1.951).
  (référence 6.649 ; meilleure précédente 3.131 = FA 2.003, Miss 1.129)
- Interprétation : non concluante, gain faible. -0.049 pt (3.082) ou -0.061 pt (3.070 avec d_end -0.05), du même ordre que le
  bruit entre modèles (~0.05 pt). Le diagnostic collar montre que l'erreur dans la bande ±50 ms n'a PRESQUE PAS bougé
  (0.677 pt contre 0.70 pt). La résolution temporelle n'était donc pas le facteur limitant. Supposé : une bonne part de cette
  bande vient de l'imprécision des frontières de l'annotation elle-même, et elle est difficile à réduire.
- Décision : garder provisoirement (`vad_fine`, seuil 0.25, d_end 0 ; je ne garde pas d_end -0.05, gain de 0.012 pt = bruit).
  Meilleure : 3.082 %.
- Remise en question : SIGNALEMENT DE PLATEAU. Depuis l'itération 12, chaque changement rapporte <= 0.05-0.07 pt (12 : -0.165 ;
  14 : -0.022 ; 15 : -0.049). Les post-traitements (4, 5, 6, 10, 13, 14) sont épuisés. L'erreur restante se partage entre
  ~0.7 pt dans ±50 ms des frontières (insensible à la résolution) et ~1.35 pt de régions (collar ±0.2 s), dont une part de pauses
  < 0.4 s. Dernier axe non exploré : des caractéristiques qui séparent la parole de la musique (modulation 2-8 Hz, Scheirer-Slaney).

### Itération 16 — caractéristiques de modulation 2-8 Hz (feats1="base+local+mod")
- Réflexion : régions d'erreur hors frontières (~1.35 pt au collar ±0.2 s) : musique/foule acceptées (iwdjy, lknjp : longs FA
  périodiques à log-odds confiants), parole sur fond sonore manquée.
- Hypothèse : si j'ajoute au 1er étage l'énergie de modulation 2-8 Hz des enveloppes de 8 groupes de bandes mel (moyenne sur 1 s,
  absolue et relative à la variance de l'enveloppe), alors le FA sur musique/bruit stationnaires doit baisser, parce que la parole
  est modulée à ~4 Hz (syllabes) et la musique/la foule beaucoup moins.
- Changement : `build_X(... "+mod")` (+18 dimensions). Mesuré sur le 1er étage seul, puis dans toute la chaîne `vad_fine(feats1=...)`
  (nouvelles prédictions hors-échantillon + nouveau 2e étage). Balayage seuil ∈ {0, 0.25, 0.5, 0.75}.
- Commandes : `python vad.py eval --method model --set kind=hgb feats=base+local+mod seuil=1 d_end=-0.1` ;
  `python vad.py eval --method fine --set feats1=base+local+mod --sweep seuil=0,0.25,0.5,0.75`
- Résultats :
  1er étage seul (seuil 1, d_end -0.1) : FA 2.206 % | Miss 1.034 % | critère 3.239 %  (sans mod : 2.320/0.976/3.296)
  chaîne fine : 0 : 2.345/0.764/3.109 ; 0.25 : FA 2.107 % | Miss 0.949 % | critère 3.055 % ; 0.5 : 1.917/1.185/3.102 ; 0.75 : 1.735/1.433/3.167
  (meilleure précédente 3.082 = FA 2.131, Miss 0.951)
- Interprétation : non concluante. 1er étage : -0.057 pt (FA -0.114, Miss +0.058) ; chaîne complète : -0.027 pt. Les deux écarts
  sont au niveau du bruit entre modèles (~0.05 pt). Le 2e étage récupérait sans doute déjà une partie de cette information.
- Décision : pas adopté seul. Conservé comme membre d'ensemble (itération 17).

### Itération 17 — ensemble de deux chaînes fines (moyenne des log-odds)
- Hypothèse : si je fais la moyenne des log-odds 10 ms des deux chaînes (1er étage base+local et base+local+mod), alors la variance
  de modèle (~0.05 pt) baisse et le critère aussi, parce que les erreurs des deux modèles ne sont pas parfaitement corrélées.
- Changement : `vad_fine_ens(members=("base+local", "base+local+mod"))`, balayage seuil ∈ {0, 0.25, 0.5}.
- Commande : `python vad.py eval --method fine_ens --sweep seuil=0,0.25,0.5`
- Résultats : 0 : 2.261/0.788/3.049 ; 0.25 : FA 2.042 % | Miss 0.991 % | critère 3.032 % ; 0.5 : 1.823/1.246/3.069
  (référence 6.649 ; meilleure précédente 3.082)
- Interprétation : non concluante, gain faible. -0.050 pt contre vad_fine. C'est le sens attendu d'un ensemble, mais l'écart
  reste au niveau du bruit.
- Décision : retenue comme meilleure version mesurée (3.032 %), avec la réserve que les itérations 15 à 17 ne rapportent
  ensemble que -0.099 pt (3.131 -> 3.032), chaque pas étant indistinguable du bruit.

### Arrêt sur plateau (critère 3 % NON atteint)
- Je signale que l'approche a atteint son plateau : sur les 6 dernières itérations (12-17), une seule (12, stacking, -0.165 pt)
  a un gain nettement au-dessus du bruit. Les autres font chacune <= 0.05 pt ou se dégradent.
- Raisons observées :
  (a) environ 0.68-0.70 pt d'erreur se trouvent à ±50 ms des frontières de référence et ne bougent pas quand je passe de
      32.5 ms à 10 ms (itération 15) ;
  (b) le comblement des pauses < 0.5 s coûte 0.94 pt même avec un oracle, mais aucun classifieur testé ne localise les pauses
      réelles assez bien pour réduire min_silence (itérations 4, 6, 13).
- Continuer à balayer sur sample90 pour gagner les 0.032 pt restants reviendrait à surajuster ce sample (beaucoup de comparaisons
  à ~0.03-0.05 pt). Je m'arrête et je valide UNE version, choisie AVANT de regarder le test :
  `vad_fine_ens`, members (base+local, base+local+mod), seuil 0.25, min_silence 0.5, min_speech 0.25.

### Validation finale n° 2 (test, samples/test/sample90_seed130.rttm) — une seule exécution, version choisie à l'avance, aucun réglage
- Rappel : le test avait déjà été regardé une fois (validation de gmm_vad, 6.622 %). Aucun choix des itérations 3 à 17 ne s'est
  appuyé sur le test.
- Commandes : `python -c "import vad; vad.build_fine_cache('test','samples/test/sample90_seed130.rttm',3)"` ;
  `python vad.py eval --method fine_ens --split test --rttm samples/test/sample90_seed130.rttm --set split=test seuil=0.25 --per-file val_fine_ens_test90.csv`
- Résultat : FA 3.370 % | Miss 2.106 % | critère 5.476 % (FA 1936.6 s, Miss 1210.5 s, parole 57473.3 s)
  (référence energy_vad : FA 7.658 | Miss 3.387 | 11.045 ; gmm_vad : FA 4.100 | Miss 2.522 | 6.622)
- Par fichier : 83 fichiers améliorés / 7 dégradés contre la référence ; 79 / 11 contre gmm_vad. Erreur médiane par fichier 3.876 %.
  Pires : wprog (FA 22.6 s, Miss 262.5 s ; la GMM manquait 101.2 s, la référence 50.2 s : forte dégradation du Miss), wemos (56.2 / 70.0),
  diysk (FA 100.8), bgvvt (54.4 / 49.8), uicid (Miss 75.5). Sans les 5 pires fichiers : 4.607 %.
- Écart réglage/test : 3.032 % contre 5.476 %. Il existait déjà pour la référence (6.649 % contre 11.045 %) : le test est plus
  difficile. Gain relatif contre la référence : -54.4 % sur le réglage, -50.4 % sur le test (proches). Contre gmm_vad : -31.3 %
  sur le réglage mais seulement -17.3 % sur le test, donc une partie des gains des itérations 3-17 ne se transfère pas. Supposé, non
  vérifié : wprog contient de la parole dans des conditions absentes du dev-train (musique forte sous la voix ?) ; je n'ai pas
  écouté le fichier.

---
## Phase 3 (demande de l'utilisateur) — repartir de l'itération 12

**Nouvelle règle de décision** : une amélioration coûteuse en calcul ou en complexité qui ne rapporte qu'un gain très faible
(<= ~0.05 pt, soit le niveau du bruit entre modèles) N'EST PAS retenue. En conséquence, les itérations 15 (grille 10 ms),
16 (modulation) et 17 (ensemble) sont écartées. La meilleure VAD redevient `vad_stack` (itération 12 : 1er étage HGB base+local,
2e étage sur le contexte des log-odds, seuil 0.5, d_end -0.1) : FA 2.003 % | Miss 1.129 % | critère 3.131 % sur sample90.
Dans `vad.py`, `best_vad(y)` et `METHODS["best"]` pointent désormais sur `vad_stack` (`BEST_METHOD`, `BEST_PARAMS`). Le code des
autres versions reste en place (`best_vad_fine_ens`, `vad_fine`, `vad_fine_ens`).
Contrôle : `python vad.py eval --method best` -> FA 2.003 % | Miss 1.129 % | critère 3.131 % (FA 533.9 s, Miss 300.9 s).
Pistes imposées, dans l'ordre : (1) lissage HMM/Viterbi ; (2) augmentation parole + musique/bruit (données locales uniquement) ;
(3) réentraînement sur les 216 fichiers dev, réglages figés avant, évaluation sur le test seulement.

### Itération 18 — lissage HMM/Viterbi à 2 états (remplace min_silence / min_speech)
- Réflexion : le comblement fixe à 0.5 s bouche aussi les vraies pauses (plancher oracle 0.94 pt) et garde les faux trous de
  plus de 0.5 s. Un Viterbi pèse l'évidence accumulée : une pause courte mais nette (log-odds très négatifs) survit, un
  creux faible est comblé.
- Hypothèse : si je remplace min_silence/min_speech par un Viterbi à 2 états (émission = log-odds du 2e étage − biais,
  pénalité de transition pen = −log P), alors le FA dû aux pauses comblées baisse sans hausse du Miss.
- Changement : `vad_stack_hmm` (Viterbi numba `_viterbi_mindur`, d_end -0.1 conservé). Balayage pen ∈ {2, 4, 6, 8, 12, 16, 24}
  (biais 0.5), puis biais ∈ {0, 0.25, 0.75, 1} pour pen 4 et 8.
- Commandes : `python vad.py eval --method stack_hmm --set bias=0.5 --sweep pen=2,4,6,8,12,16,24` ;
  `for p in 4 8; do python vad.py eval --method stack_hmm --set pen=$p --sweep bias=0,0.25,0.75,1; done`
- Résultats (FA/Miss/critère %) :
  biais 0.5 : pen 2 : 1.747/1.491/3.238 ; 4 : 1.814/1.398/3.212 ; 6 : 1.890/1.338/3.228 ; 8 : 1.975/1.257/3.233 ; 12 : 2.148/1.099/3.247 ;
              16 : 2.312/0.996/3.308 ; 24 : 2.615/0.832/3.447
  pen 4 : biais 0 : 2.242/0.930/3.172 ; 0.25 : FA 2.021 % | Miss 1.146 % | critère 3.167 % ; 0.75 : 1.622/1.704/3.326 ; 1 : 1.463/2.104/3.567
  pen 8 : biais 0 : 2.415/0.819/3.234 ; 0.25 : 2.175/1.003/3.177 ; 0.75 : 1.773/1.559/3.332 ; 1 : 1.581/1.856/3.438
  (référence 6.649 ; meilleure it.12 : FA 2.003 | Miss 1.129 | 3.131)
- Interprétation : réfutée. Meilleur point 3.167 %, soit +0.036 pt (moins bon). Observé : les log-odds du 2e étage sont très
  tranchés (|llr| : p10 2.19, p25 3.82, médiane 5.83, p75 7.87 sur 20 fichiers). Le Viterbi ne fait donc que du seuillage, à peu
  près, et ne retrouve pas l'effet utile du comblement à 0.5 s.
- Décision : abandonner.

### Itération 19 — HMM à durées minimales (états chaînés) et pénalités asymétriques
- Hypothèse : si j'impose des durées minimales (non-parole >= min_sil_s, parole >= min_speech_s) par des états chaînés, ou des
  pénalités différentes pour entrer et sortir de la parole, alors le HMM retrouve au moins le comportement de min_silence et passe
  sous 3.131 %.
- Changement : pen 4, biais 0.25 ; min_sil_s ∈ {0.1, 0.2, 0.3, 0.5} ; min_speech_s ∈ {0.1, 0.25, 0.5} ; pen_ns ∈ {1, 2, 8, 12}.
- Commandes : `python vad.py eval --method stack_hmm --set pen=4 bias=0.25 --sweep min_sil_s=0.1,0.2,0.3,0.5` (idem min_speech_s, pen_ns)
- Résultats : min_sil_s 0.1 : 3.167 ; 0.2 : 3.168 ; 0.3 : 3.171 ; 0.5 : FA 2.031 % | Miss 1.149 % | 3.180 %
  min_speech_s 0.1 : 3.167 ; 0.25 : 3.167 ; 0.5 : FA 2.021 % | Miss 1.145 % | 3.166 %
  pen_ns 1 : 1.964/1.209/3.173 ; 2 : 1.986/1.191/3.177 ; 8 : 2.095/1.081/3.176 ; 12 : 2.177/1.002/3.179
- Interprétation : réfutée. Les durées minimales ne changent presque rien (<= 0.013 pt) : avec des log-odds aussi tranchés,
  la contrainte de durée ne s'active quasiment jamais. Supposé : le comblement à 0.5 s gagne surtout sur des creux où
  les log-odds sont francs mais FAUX (le 2e étage se trompe avec assurance), ce qu'aucun lissage fondé sur ces mêmes log-odds
  ne peut détecter.
- Coût : Viterbi numba, ~24 s par balayage de 7 valeurs (90 fichiers), négligeable. Complexité ajoutée, mais aucun gain.
- Décision : abandonner la piste 1 ; le post-traitement min_silence 0.5 / min_speech 0.25 reste.

### Itération 20 — augmentation : parole + non-parole énergétique (musique / bruit) à SNR variés
- Réflexion : à la validation n° 2, le pire fichier test (wprog) est surtout du Miss (262.5 s), et je suppose de la parole sur
  fond musical. Ces conditions sont rares dans dev-train : le classifieur a appris « fond musical = non-parole ».
- Hypothèse : si j'ajoute à l'entraînement des pseudo-fichiers où de la parole de dev-train est mélangée à de la non-parole
  énergétique d'autres fichiers (0/5/10/20 dB), alors le Miss sur la parole bruitée doit baisser sans hausse du FA, parce que le
  modèle voit des exemples « parole + musique = parole ».
- Changement (un seul : les données d'entraînement) : `build_aug` et `train="devtrain+aug"` pour les deux étages de `vad_stack`.
  Banque de bruit par pli = segments non-parole de la référence >= 0.5 s ET RMS >= niveau de parole du fichier - 30 dB
  (pli 0 : 423.3 s, pli 1 : 918.5 s, pli 2 : 880.9 s). Un pseudo-fichier par fichier dev-train : 120 s (fenêtre tirée au
  hasard) + bruit du MÊME pli (pas de fuite entre plis du hors-échantillon), SNR tiré dans {0, 5, 10, 20} dB (34/26/34/32 fichiers).
  126 pseudo-fichiers, durée moyenne 113.3 s (3.97 h), 93.2 % de trames parole. Prédictions hors-échantillon par pli de la
  parole source. Balayage seuil ∈ {0, 0.25, 0.5, 0.75, 1}, d_end -0.1.
- Commandes : `python -c "import vad; vad.build_aug('devtrain')"` ;
  `python vad.py eval --method stack --set train=devtrain+aug d_end=-0.1 --sweep seuil=0,0.25,0.5,0.75,1`
- Résultats :
  0 : 2.704/0.591/3.295 ; 0.25 : 2.439/0.763/3.202 ; 0.5 : 2.188/0.961/3.148 ;
  0.75 : FA 1.960 % | Miss 1.176 % | critère 3.135 % ; 1 : 1.770/1.493/3.262
  (référence 6.649 ; meilleure it.12 : FA 2.003 | Miss 1.129 | 3.131)
- Par fichier (seuil 0.75 contre it.12) : 47 fichiers améliorés, 42 dégradés. Gains : onpra -7.0 s (Miss 13.6 -> 5.7), nrogz -5.6 s
  (Miss 14.9 -> 3.1, FA 18.2 -> 24.4). Pertes : bkwns +9.5 s (FA 8.9 -> 18.4), kkghn +4.5 s (Miss 21.5 -> 27.8).
- Coût : génération 47 s (banque + 126 pseudo-fichiers, 80 Mo de cache + 136 Mo de banque) ; entraînement des deux étages avec
  plis : 3 min 10 s, contre ~1 min 45 s sans augmentation ; un pipeline de données en plus.
- Interprétation : non concluante sur le réglage (+0.004 pt). Les gains et les pertes par fichier se compensent. Limite
  importante : sample90 contient sans doute peu de parole sur musique, donc ce sample ne peut pas mesurer l'effet visé. Je ne
  peux pas le mesurer sur le test sans y faire un choix, ce qui est interdit.
- Décision : non retenue (règle : coût de complexité et de calcul pour un gain nul mesuré). Le code reste dans `vad.py`.
  Piste à reprendre avec un sample de réglage riche en parole sur musique, ou avec un corpus de musique externe (MUSAN), non
  téléchargé conformément à la consigne.

### Itération 21 (piste 3) — réentraînement de la version retenue sur les 216 fichiers dev (production)
- Réglages FIGÉS AVANT (aucun réglage possible sans sample de réglage hors entraînement) : `vad_stack`, feats1 base+local, step 4,
  HGB 300 it. / 63 feuilles (les deux étages), prédictions hors-échantillon en 3 plis (sur 216 fichiers),
  seuil 0.5, min_silence 0.5, min_speech 0.25, d_end -0.1, sans augmentation (itération 20 non retenue).
- Comparaison prévue, UNE seule exécution chacune sur le test : même configuration entraînée sur dev-train (126 fichiers) contre
  sur all-dev (216 fichiers) (`train=alldev`). sample90 n'est pas évaluable pour la version all-dev (fichiers vus à l'entraînement).
- Commandes : entraînement `python -c "import vad; vad.get_stack_model('base+local', 4, 'alldev'); vad.get_model('hgb','base+local',4,train='alldev')"` (203 s) ;
  `python vad.py eval --method stack --split test --rttm samples/test/sample90_seed130.rttm --set seuil=0.5 d_end=-0.1 --per-file val_it12_devtrain_test90.csv`
  `python vad.py eval --method stack --split test --rttm samples/test/sample90_seed130.rttm --set seuil=0.5 d_end=-0.1 train=alldev --per-file val_it21_alldev_test90.csv`
- Résultats (test sample90, 57473.3 s de parole ; une exécution chacune) :
  it.12 entraînée sur dev-train (126 fichiers) : FA 3.371 % | Miss 2.390 % | critère 5.761 % (FA 1937.3 s, Miss 1373.5 s)
  it.12 entraînée sur all-dev (216 fichiers)   : FA 3.459 % | Miss 1.976 % | critère 5.434 % (FA 1987.7 s, Miss 1135.5 s)
  (référence energy_vad : FA 7.658 | Miss 3.387 | 11.045 ; pour mémoire, gmm_vad 6.622 et vad_fine_ens écartée 5.476)
- Par fichier : dev-train contre référence : 81 fichiers améliorés / 9 dégradés. All-dev contre dev-train : 52 améliorés / 37 dégradés.
  Pires fichiers (all-dev) : wprog (FA 45.7 s, Miss 170.9 s ; dev-train : 27.0 / 272.4), wemos (58.1 / 61.2 ; dev-train : 44.9 / 104.9),
  uicid (20.9 / 95.1), diysk (99.4 / 12.0), bgvvt (64.1 / 31.9). Sans les 5 pires fichiers : 4.671 %.
- Interprétation : réentraîner sur 216 fichiers fait baisser le Miss (2.390 -> 1.976 %) et monter un peu le FA
  (3.371 -> 3.459 %) : -0.327 pt (-5.7 % relatif), surtout sur les fichiers à Miss (wprog -101.5 s de Miss, wemos -43.7 s).
  C'est une seule comparaison sur un seul tirage d'entraînement : cohérent avec « plus de données aide », mais sans intervalle
  de confiance. La décision d'utiliser all-dev en production avait été fixée AVANT cette mesure (consigne de la piste 3) ; la
  mesure ne sert pas à choisir. Le seuil (0.5) a été réglé pour des modèles dev-train : pour all-dev, il n'a pas été réajusté.
- Coût : entraînement 203 s (contre ~105 s sur dev-train), mêmes modèles à l'inférence (coût identique).
- Décision : `best_vad(y)` (production) utilise les modèles all-dev (`PROD_PARAMS`) ; `METHODS["best"]` reste la version dev-train.

### Arrêt (consigne de l'utilisateur : fin après l'itération 21)
- Critère 3 % non atteint. Meilleure version évaluable sur le réglage : it.12 `vad_stack`, 3.131 % (FA 2.003, Miss 1.129).
- Phase 3 : pistes 1 (HMM, it. 18-19) et 2 (augmentation, it. 20) réfutées ou non concluantes ; piste 3 (all-dev) : 5.761 -> 5.434 % sur le test.

---
## Nettoyage (demande de l'utilisateur, après l'itération 21)
Version conservée : it.12 `vad_stack` (feats1 base+local, seuil 0.5, d_end -0.1, min_silence 0.5, min_speech 0.25).
Production : `best_vad(y)`, modèles all-dev (`PROD_PARAMS`). Évaluation : `python vad.py eval --method best`, modèles dev-train (`BEST_PARAMS`).

**vad.py** : 1103 -> 651 lignes (version d'origine au commit HEAD : 205 lignes). Partie d'origine (main, energy_vad locale, trace,
hysteresis...) inchangée. Blocs conservés recopiés à l'identique (vérification octet par octet) : import_energy_vad_ref, periodicity,
compute_features, _cache_one, build_cache, load_cache, load_ref, frames_to_timeline, adjust_bounds, vad_reference, vad_energy,
dev_train_uris, _mstd, stack_X, evaluate, fmt, _parse_value. build_X réduit aux options base / base+local, avec une matrice
identique à l'ancienne (np.array_equal sur hgdez et ahnss). get_model, speech_llr, oof_llr, get_stack_model, vad_stack et cli
sont simplifiés, avec des noms de fichiers de modèles inchangés.
Fonctions retirées : gmm_vad, vad_model, file_adapted_llr, vad_fine, vad_fine_ens, best_vad_fine_ens, fine_X, fine_labels,
get_fine_model, load_fine, fine_energy, _fine_one, build_fine_cache, refine_bounds, vad_stack_refined, _viterbi_mindur_impl,
hmm_smooth, stack_llr, vad_stack_hmm, build_noise_bank, _aug_one, build_aug, fold_of, item_source. Retirés aussi : options
feats "mfcc", "+mod", "+long" ; kind="gmm" ; paramètres adapt*, smooth_frames, d_start, train="devtrain+aug" ; METHODS model,
stack_refined, fine, fine_ens, stack_hmm ; constantes FINE_*, AUG_*, BEST_METHOD ; construction du cache fin dans `cache`.

**results/vad_agent/ : 1.1 Go (1122755981 octets) -> 779 Mo (815448542 octets).**
Supprimés :
- CSV de versions non retenues : it10_best_sample90.csv, it17_fine_ens_sample90.csv, it20_aug_sample90.csv,
  it2_gmm_sample90.csv, it2_gmm_seuil0.25_sample90.csv, it3_hgb_sample90.csv, it7_hgb_local_sample90.csv,
  val_fine_ens_test90.csv, val_gmm_test90.csv ;
- journaux de cache : cache_dev.log, cache_test.log ;
- modèles abandonnés : fine_hgb_base+local+mod_s12.pkl, fine_hgb_base+local_s12.pkl, gmm_mfcc_s4.pkl,
  hgb_base+local+long_s4.pkl, hgb_base+local+mod_s4.pkl, hgb_base+local_s1.pkl, hgb_base+local_s4_traindevtrain+aug.pkl,
  hgb_base_s4.pkl, oof_hgb_base+local+mod_f3_s4.npz, oof_hgb_base+local_f3_s4_devtrain+aug.npz,
  stack_hgb_base+local_s4_devtrain+aug.pkl ;
- dossiers : cache_fine/ (53 Mo), aug/ (136 Mo, banques de bruit), cache/aug/ (80 Mo).
Gardés :
- journal.md, rapport.md ;
- CSV : it0_reference_sample90.csv, it12_best_sample90.csv, val_reference_test90.csv, val_it12_devtrain_test90.csv,
  val_it21_alldev_test90.csv ;
- models/ : hgb_base+local_s4.pkl, stack_hgb_base+local_s4.pkl, oof_hgb_base+local_f3_s4.npz (dev-train) ;
  hgb_base+local_s4_trainalldev.pkl, stack_hgb_base+local_s4_alldev.pkl, oof_hgb_base+local_f3_s4_alldev.npz (all-dev).
  Les .npz hors-échantillon ne servent qu'à réentraîner le 2e étage (14 Mo) ; je les garde pour pouvoir reproduire.
- cache/dev (405 Mo : caractéristiques + sortie energy_vad, indispensables pour évaluer et réentraîner) et cache/test (355 Mo).

Vérifications après nettoyage :
- `python vad.py eval --method best` -> FA 2.003 % | Miss 1.129 % | critère 3.131 % (FA 533.9 s, Miss 300.9 s), identique ;
  aucun modèle réentraîné (dates des fichiers de models/ inchangées).
- `python vad.py eval --method reference` -> FA 4.177 % | Miss 2.472 % | critère 6.649 %, identique.
- `best_vad(y)` sur test/wprog : 319 segments, liste identique à celle d'avant le nettoyage.
- `python vad.py` (main d'origine), lancé depuis un dossier temporaire (liens vers data/ et samples/) pour ne pas réécrire
  figures/ : exit 0, 1 min 49 s ; le CSV produit est identique à figures/vad_delta.csv (suivi par git).
- `git diff --stat -- vad.py` après nettoyage : +446 / -1.
