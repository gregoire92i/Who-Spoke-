# Guide de lecture des figures exploratoires : k oracle vs eigengap

Dossier : `results/diag_k_oracle_vs_eigengap/figures_exploration/`. Une sous-section par piste.

**Protocole commun.** Dev sample 90 (`samples/dev/sample90_seed130.rttm`), test non utilisé.
Le pipeline est celui de `HEAD:baseline_oracle.py` : frames 65 / 32,5 ms, 13 MFCC, windows 2 s / hop 1 s / min 0,5 s, mean_std.
Le clustering `spectral_clustering` garde ses défauts (p_neighbors=0.1, min_neighbors=5, max_speakers=10, KMeans seed 42),
identiques dans les deux bras. Le seul changement : k = eigengap vs k = n_ref.
Sauf mention contraire, la VAD est **oracle**. Les autres VAD sont energy_vad (version actuelle) et best_vad avec `train="devtrain"`
(sans fuite). Évaluation : collar 0, UEM [0, durée wav], micro.

**Vocabulaire des figures**
- **Régime** : erreur de l'eigengap sur le fichier. Couleurs fixes : orange `k_eig<n_ref`, gris `k_eig=n_ref`,
  bleu `k_eig>n_ref`. Le gris est le témoin : k ne change pas, donc Δ = 0 par construction.
- **Δ** : valeur avec k oracle moins valeur avec eigengap. Δ > 0 = l'oracle dégrade.
- **Fichiers « cas »** : 4 où l'oracle dégrade (kdfqk, bauzd, uvnmy, evtyi), 2 où il aide (hkzpa, hgdez),
  2 où l'eigengap surestime (oxxwk, zfkap), 2 où k_eig est correct (odkzj, qygfk).
- **Window pure** : son locuteur majoritaire couvre au moins 80 % de la window, sans overlap. Une window **mixte** ne l'est pas.
- **Types de confusion** (décompte trame par trame, écart à pyannote inférieur à 5 s sur 6 000 s) :
  - `impurity` : erreur dans un cluster apparié à son locuteur majoritaire ;
  - `mismapped` : cluster apparié par `optimal_mapping` à un autre locuteur que son majoritaire ;
  - `unmapped` : cluster associé à aucun locuteur.
- **« Regroupé »** : contrôle oracle. Chaque cluster est fusionné sur son locuteur majoritaire avant le calcul du DER.
- **Trame disputée** : couverte par des windows de labels différents. C'est là que le vote du post-traitement décide.

---

## p01 : répartition de la hausse
**Question** : la hausse vient-elle de tous les fichiers ou de quelques-uns, et de quel type de fichier ?

- **`delta_conf_trie_cumule_vad_{oracle,energy,stack_devtrain}.png`**
  - Ce qu'elle montre : une barre par fichier = sa contribution à Δconfusion micro (pts de DER), triée ; courbe noire = cumul.
  - Si la hausse est concentrée : la courbe atteint le total en quelques fichiers, puis reste plate.
  - Si elle est diffuse : la courbe monte régulièrement.
  - Les barres négatives sont les fichiers où l'oracle aide.
- **`dDER_vs_caracteristiques_vad_oracle.png`** et **`dconf_s_vs_caracteristiques_vad_oracle.png`**
  - Ce qu'elles montrent : 9 nuages ΔDER (ou Δconf en s) en fonction de n_ref, k_eig − n_ref, durée, part du dominant,
    nombre de locuteurs avec moins de p windows, taux d'overlap, durée de parole, durée moyenne d'un tour, n windows.
  - Le titre donne le ρ de Spearman. Ce n'est qu'une corrélation, elle ne suffit pas à conclure à une cause.
  - Si une caractéristique explique la hausse : nuage monotone, ρ élevé, et c'est vrai aussi dans le seul régime orange.
  - Sinon : nuage sans forme, ou forme due uniquement au mélange des régimes.
- **`dDER_vs_caracteristiques_regime_sous_estime.png`**
  - Ce qu'elle montre : les mêmes nuages, restreints aux fichiers où l'eigengap sous-estime k.
  - À utiliser pour voir si une corrélation survit à l'intérieur d'un régime.

## p02 : k
**Question** : l'eigengap se trompe-t-il, et le meilleur k est-il n_ref ?

- **`k_eig_best_k_vs_n_ref.png`**
  - Gauche, heatmap k_eig × n_ref : les cases sous la diagonale rouge sont les sous-estimations.
  - Milieu, meilleur k (DER minimal pour k ≤ n_ref+3) × n_ref : si n_ref est le bon k pour ce pipeline, tout est sur la diagonale.
  - Droite : k_eig − n_ref vs meilleur k − n_ref, couleur = ΔDER.
    - Points sur la diagonale : l'eigengap trouve le meilleur k.
    - Points rouges loin de la diagonale : fichiers où imposer n_ref est coûteux.
- **`courbes_metriques_vs_k_fichiers_representatifs.png`**
  - Ce qu'elle montre : pour les 10 fichiers cas, DER, confusion/parole, JER, purity et coverage en fonction de k.
    Ligne grise = k_eig, ligne verte = n_ref.
  - Morcellement : la coverage chute quand k augmente, sans gain de purity.
  - Isolement de nouveaux locuteurs : la purity monte, le JER baisse et le DER baisse aussi.
  - Comparer la valeur du DER à la ligne grise et à la ligne verte.
- **`courbes_relatives_tous_fichiers.png`**
  - Ce qu'elle montre : DER(k) − DER(n_ref) et coverage(k) − coverage(n_ref), un trait par fichier.
  - Si un k plus petit est uniformément meilleur : la plupart des traits sont sous 0 à gauche de 0.

## p03 : matrices de confusion cluster × locuteur (par fichier cas)
- **`<uri>_confusion_eig_vs_oracle.png`**
  - Lignes = locuteurs (temps total) ; colonnes = clusters ; cases = secondes.
  - Rouge gras : case retenue par l'appariement du DER.
  - `c→∅` en rouge : cluster non apparié. `✗` : locuteur non apparié.
  - Morcellement : avec le k oracle, une ligne (souvent le dominant) se répartit sur plusieurs colonnes, dont des colonnes
    non appariées ou appariées à un autre locuteur.
  - Isolement réussi : chaque nouvelle colonne a un seul gros bloc sur une ligne différente.
  - Fusion avec l'eigengap : une colonne avec plusieurs gros blocs.

## p04 : timelines (par fichier cas)
- **`<uri>_timeline.png`** (de haut en bas)
  - une ligne par locuteur de référence ;
  - l'overlap de la référence ;
  - la parole manquée par best_vad devtrain (pour situer ce qu'une VAD réelle retire) ;
  - la parole non couverte par une window (segments < 0,5 s) ;
  - l'hypothèse eigengap et ses trames en confusion (rouge) ;
  - l'hypothèse k oracle et ses trames en confusion (rouge).
- Comment la lire :
  - Les zones rouges ajoutées par l'oracle sont-elles dans des tours longs d'un même locuteur (alternance de clusters à
    l'intérieur d'un tour) ?
  - Ou près des changements de locuteur, dans l'overlap, ou là où la VAD retire de la parole ?

## p05 : embeddings
- **`<uri>_projections.png`** : PCA et t-SNE (seed 0 ; UMAP non installé). Gauche = vrai locuteur (gris = mixte),
  milieu = clusters eigengap, droite = clusters k oracle.
  - Locuteurs séparables : les couleurs de la référence forment des amas distincts, et le k oracle les retrouve.
  - Locuteurs non séparables : un petit locuteur est noyé dans un amas, et le k oracle découpe l'amas en régions qui ne
    suivent pas les couleurs de la référence.
  - t-SNE déforme les distances globales : lire les voisinages, pas la distance entre amas.
- **`<uri>_cosinus_intra_inter.png`** : histogrammes des cosinus entre windows pures du fichier, même locuteur vs différents.
  Plus ils se recouvrent, moins les locuteurs sont séparables avant tout clustering.
- **`global_qualite_embeddings_vs_delta.png`**
  - Histogramme poolé des cosinus intra / inter.
  - Nuages AUC, silhouette réf., cos intra moyen et cos inter moyen vs Δconfusion.
  - Silhouette des partitions eigengap vs k oracle.
  - Si des embeddings médiocres expliquent la hausse : Δconf augmente quand l'AUC ou la silhouette baissent.
- **`global_fisher_par_dimension.png`** : ratio de Fisher (inter / intra) par dimension (μc0..12, σc0..12), par fichier.
  Montre quelles dimensions portent l'identité du locuteur.

## p06 : graphe d'affinité et spectre
- **`<uri>_affinite_spectre.png`**
  - Gauche : cosinus brut, windows triées par locuteur puis temps (lignes vertes = changements de locuteur).
  - Milieu : `A_bin` après élagage à p voisins et symétrisation (ce que voit le laplacien).
  - Droite : 25 premières valeurs propres, avec la coupure eigengap (gris) et n_ref (vert).
  - Si un locuteur a son bloc dans `A_bin` : carré noir sur la diagonale, et une valeur propre proche de 0 en plus.
  - Si son bloc a disparu à l'élagage : présent (même faiblement) dans le cosinus brut, absent de `A_bin`.
  - Si le petit locuteur n'est même pas visible dans le cosinus brut : le problème est en amont (embeddings).
- **`global_spectre_vs_delta.png`** : ΔDER vs λ(n_ref+1) − λ(n_ref), λ(n_ref), nombre de valeurs propres < 0,05, et l'écart
  retenu par l'eigengap. Si le spectre prédit la dégradation, ΔDER est fort quand l'écart à n_ref est faible.

## p07 : locuteurs rares
- **`global_locuteurs_rares.png`**
  - (a) Chaque locuteur : windows/p vs temps de parole. Couleur = isolé (majoritaire d'au moins un cluster) avec l'eigengap,
    avec l'oracle, les deux, ou aucun.
  - (b) Rappel par locuteur sous l'appariement du DER, eigengap vs oracle (orange = dominant). Sous la diagonale, l'oracle
    retire du temps à ce locuteur.
  - (c) Temps des locuteurs non appariés par fichier.
  - (d) Nombre de locuteurs non appariés par fichier.
  - Si la taille relative à p joue : le basculement isolé / non isolé se fait vers windows/p ≈ 1.
- **`global_windows_par_locuteur_selon_vad.png`** : fonction de répartition de windows/p pour les 3 VAD. Si la VAD retirait
  les windows des petits locuteurs, les courbes seraient décalées.
- Figure de la première analyse : `../figures/locuteurs_rares_isolation.png`.

## p08 : overlap, frontières, vote
- **`global_overlap_frontieres_vad_{oracle,stack_devtrain}.png`**
  - Confusion en secondes puis en taux : dans l'overlap, à 0-0,5 / 0,5-1 / 1-2 / > 2 s d'une frontière de tour.
  - Taux de confusion dans les trames disputées et non disputées.
  - Si la hausse vient des frontières ou de la taille des windows (2 s) : le Δ est concentré à moins de 1 s des frontières
    ou dans l'overlap.
  - Si elle vient de l'intérieur des tours : le Δ est surtout à > 2 s.
  - La part de trames disputées (légende) montre le travail du vote du post-traitement.
- **`par_fichier_delta_par_zone.png`** : par fichier, Δconfusion de chaque zone vs Δconfusion totale (diagonale = tout le Δ
  du fichier dans cette zone). Le titre donne la part du Δ total.

## p09 : VAD
- **`global_vad.png`** :
  - DER et confusion micro, eigengap vs oracle, pour les 3 VAD ;
  - Δconfusion par fichier, VAD oracle vs energy, et VAD oracle vs stack devtrain.
  - Si la VAD n'est pas en jeu : écart semblable pour les 3 VAD, points sur la diagonale.
  - Si elle l'est : écart très différent selon la VAD, points hors diagonale.
- **`global_types_confusion_par_vad.png`** : impurity / mismapped / unmapped pour les 3 VAD et les 2 bras.

## p10 : sensibilité
- **`global_seeds_p_neighbors.png`**
  - Gauche : DER et confusion pour 5 seeds KMeans (0, 1, 2, 3, 42).
  - Milieu : ΔDER par fichier, médiane et min-max sur les 5 seeds.
  - Droite : balayage p_neighbors ∈ {0.02, 0.05, 0.1, 0.15, 0.2} (seed 42).
  - Si l'écart est du bruit : il change de signe ou d'ampleur avec la seed.
  - Si l'écart dépend du graphe kNN : il varie avec p.
- **`par_fichier_delta_selon_p.png`** : distribution par fichier de ΔDER pour chaque p.

## p11 : métrique
- **`global_metrique_decomposition.png`**
  - Gauche : DER, DER forgiving, JER et confusion pour eigengap, k oracle et leurs versions « regroupées ».
  - Milieu : types de confusion par régime.
  - Droite : ΔDER vs ΔJER par fichier.
  - Si la hausse est un artefact de l'appariement one-to-one : la hausse est surtout mismapped / unmapped, et la version
    regroupée de l'oracle n'est pas pire que celle de l'eigengap.
  - Si la partition oracle est intrinsèquement moins bonne : la hausse est en impurity, et l'oracle regroupé reste pire.
  - DER et JER pondèrent différemment le temps et les locuteurs (JER = moyenne par locuteur). Leur désaccord dit quel
    critère est touché.

## Pistes non couvertes par ces figures
- Écoute des fichiers (musique, bruit) : non faite.
- Autre type d'embedding ou autre clustering (AHC) : non testé (ce serait une modification du pipeline).
- Taille des windows (`win_s`) : non balayée ici. Son effet sur les frontières n'est visible qu'indirectement (p08).

---

## Observations chiffrées (Claude, séparées du guide)
Sources : `explore_numbers_sample90.log`, `aggregate_sample90.log`, `extra_stats_sample90.log`. À vérifier sur les figures.

| VAD | bras | DER | forgiving | JER | FA | miss | confusion | impurity | mismapped | unmapped |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | eigengap | 28,67 | 25,99 | 61,13 | 0,03 | 6,87 | 21,77 | 17,42 | 2,54 | 1,83 |
| oracle | k oracle | 33,06 | 30,28 | 53,46 | 0,03 | 6,87 | 26,16 | 6,02 | 10,60 | 9,55 |
| oracle | eigengap regroupé | 25,32 | 22,60 | 59,74 | | | 18,42 | | | |
| oracle | k oracle regroupé | 17,51 | 14,44 | 49,88 | | | 10,61 | | | |
| energy | eigengap / k oracle | 33,95 / 38,27 | 31,60 / 35,90 | 64,71 / 55,81 | 3,87 | 8,25 | 21,84 / 26,15 | | | |
| stack devtrain | eigengap / k oracle | 30,14 / 34,23 | 27,52 / 31,55 | 62,01 / 53,69 | 1,62 | 6,32 | 22,20 / 26,29 | | | |

- **Régimes (VAD oracle)** :
  - 50 fichiers k_eig < n_ref : +4,92 pts ;
  - 8 fichiers k_eig > n_ref : −0,53 pt ;
  - 32 fichiers k_eig = n_ref : 0.
  - Les 10 fichiers les plus dégradés font +4,74 pts sur +4,39 au total.
- **Locuteurs de référence non appariés** (somme sur 90 fichiers) : 171 (4 991 s) avec l'eigengap, 65 (563 s) avec l'oracle.
- **Zones** (VAD oracle, secondes de confusion, eigengap → oracle) :
  - overlap 231 → 319 ;
  - à < 0,5 s d'une frontière 772 → 899 ;
  - à > 2 s 3 420 → 4 118 ;
  - trames disputées 592 → 2 247 (part de la parole disputée 4,3 % → 15,0 %) ;
  - trames non disputées 5 434 → 4 994.
- **Seeds** : DER k oracle entre 32,98 et 33,07 %, eigengap entre 28,66 et 28,67 %. Sur 5 seeds, l'amplitude de ΔDER par
  fichier a une médiane de 0,00 pt et un maximum de 6,05 pts (pnook) ; 8 fichiers dépassent 1 pt.
- **p_neighbors** (DER eigengap / k oracle) : 0.02 : 25,41 / 28,31 ; 0.05 : 29,26 / 30,73 ; 0.1 : 28,67 / 33,06 ;
  0.15 : 31,70 / 34,44 ; 0.2 : 33,53 / 35,71. L'oracle est plus mauvais pour chaque p testé ; l'écart varie de 1,47 à 4,39 pts.
- **Corrélation de Spearman de ΔDER** (50 fichiers sous-estimés) : part du dominant ρ = +0,54 (p = 5e-5) ;
  n_ref ρ = −0,28 (p = 0,045) ; écart propre à n_ref ρ = +0,21 (p = 0,14).
- **Lecture de Claude** (voir `rapport.md`) : morcellement du locuteur dominant quand les petits locuteurs n'ont pas de bloc
  dans le graphe kNN. Les figures ci-dessus permettent de le confirmer ou de l'infirmer, piste par piste.
