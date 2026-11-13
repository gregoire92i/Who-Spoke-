# Pourquoi le k oracle dégrade-t-il le DER ? (dev sample 90, 13/11/2026)

## 1. Question et documents lus
Question : avec k = n_ref, DER moderne 33,49 % → 38,76 %, hausse uniquement en confusion. Pourquoi ?
Lus : `CLAUDE.md`, `docs/etat_du_projet.md`, `docs/notes-importantes-et-difficultes.md`, `docs/choix_techniques.md`
(liste des décisions, D-55 ; aucune décision D-xx sur le clustering ni sur p_neighbors). Code : `baseline.py`,
`baseline_oracle.py` (working tree et HEAD), `clustering.py`, `embeddings.py`, `post_treatment.py`, `vad.py` (best_vad).

## 2. Protocole
- 90 fichiers `samples/dev/sample90_seed130.rttm`. Test non utilisé.
- Pipeline = `HEAD:baseline_oracle.py` : frames 65 ms / hop 32,5 ms, 13 MFCC, windows 2,0 s / hop 1,0 s / min 0,5 s, mean_std,
  mode segment ; `spectral_clustering` avec ses **valeurs par défaut** (p_neighbors=0.1, min_neighbors=5, max_speakers=10,
  standardize, KMeans seed 42). t-SNE/PCA seed 0.
- VAD : **oracle** (analyse principale) ; energy_vad (version actuelle, −30 dB, 0,5 / 0,3 s) ; best_vad(train="devtrain")
  (entraînée hors sample 90, sans fuite).
- Un seul changement par comparaison : k = eigengap vs k = n_ref (mêmes windows, mêmes vecteurs propres).
- Collar 0, UEM = durée du wav, micro ; DER moderne, DER forgiving, JER, FA/miss/confusion ; purity/coverage pyannote ;
  ARI/AMI au niveau windows (label = locuteur majoritaire ; window mixte = majoritaire < 80 % ou overlap).

### Configurations des chiffres de la question (vérifié)
| Run | VAD | p / max_speakers | eigengap | k oracle |
|---|---|---|---|---|
| CSV committés (HEAD, 4ae8849) | energy_vad (ancienne version) | 0.1 / 10 | 33,49 % (conf 21,55) | 38,76 % (conf 26,82) |
| CSV working tree (12/11) | best_vad **alldev** | 0.1 / 10 | 29,71 % (conf 22,63) | 33,92 % (conf 26,84) |
| ce diagnostic | oracle | 0.1 / 10 | 28,67 % (conf 21,77) | 33,06 % (conf 26,16) |
| ce diagnostic | energy_vad actuelle | 0.1 / 10 | 33,95 % | 38,27 % |
| ce diagnostic | best_vad devtrain | 0.1 / 10 | 30,14 % | 34,23 % |
Dans chaque paire, seul k change (n_windows, n_uncovered, VAD_error identiques par fichier). Le phénomène (+4,1 à +4,4 pts
de confusion) est **le même avec les 3 VAD**.

## 3. Tableau des causes à effets
| Changement | Effet observé (VAD oracle, 90 fichiers) | Mécanisme dans le code | Preuve | Confiance |
|---|---|---|---|---|
| k eigengap → n_ref | DER 28,67 → 33,06 %, JER 61,13 → 53,46 %, purity 80,2 → 88,6 %, coverage 88,3 → 69,1 % | `clustering.py:74` (k = n_speakers) puis `:92` KMeans sur les k premiers vecteurs propres. S'il n'existe pas k blocs dans `A_bin`, KMeans coupe le plus gros bloc (le locuteur dominant) | agregats_micro, purity_coverage_eig_vers_oracle.png | forte |
| idem, confusion par type | confusion « impureté » 17,42 → 6,02 pts ; confusion « fragmentation » 4,37 → 20,15 pts | clusters supplémentaires = morceaux du dominant ; l'appariement hongrois (one-to-one) les associe à un petit locuteur ou à aucun → tout leur temps compte en confusion | agregats_micro (conf_impurity / conf_fragment) | forte |
| idem, selon l'erreur de l'eigengap | k_eig < n_ref (50 fichiers) : +4,92 pts ; k_eig > n_ref (8) : −0,53 ; k_eig = n_ref (32) : 0 | — | delta_confusion_par_fichier.png | forte |
| locuteurs rares | 172/379 locuteurs ont moins de p windows (p = 10 % des windows) ; neighbor purity 0,18 si n_win < 0,5 p | `clustering.py:56-62` : chaque window garde p voisins → un locuteur de n < p windows a au moins p − n voisins chez les autres → pas de bloc propre → pas de valeur propre ≈ 0 qui lui est associée | locuteurs_rares_isolation.png, speakers_sample90.csv | forte |
| p_neighbors 0.1 → 0.02 (diagnostic) | pénalité du k oracle +4,39 → +2,90 pts ; locuteurs isolés (k oracle) 252 → 274 | même ligne : un p plus petit laisse un bloc à des locuteurs plus petits | agregats_micro | moyenne (non monotone : p=0.05 eigengap 29,26 % > p=0.1) |
| VAD oracle / energy / stack | Δconfusion +4,39 / +4,32 / +4,09 pts | locuteurs sans window : 9 / 8 / 11 sur 379 (17 / 13 / 24 s) → négligeable | aggregate_sample90.log | forte (la VAD n'est pas la cause) |

## 4. Résultats observés / interprétation / hypothèses

### Observé
1. **Concentration** : la hausse vient des 50 fichiers où l'eigengap sous-estime k (+4,92 pts), en partie compensée par
   les 8 où il surestime (−0,53). Les 10 fichiers les plus dégradés pèsent +4,74 pts sur +4,39 au total. Exemples : kdfqk
   (20 réf., eigengap 2) +308 s de confusion, le dominant y est coupé en 13 clusters ; evtyi (2 réf., eigengap 1) +163 s ;
   bauzd +144 s. Inversement, l'oracle aide fortement sur hkzpa (−227 s), hgdez (−201 s), ldnro (−166 s), où l'eigengap donne k=1.
2. **Sur-segmentation du dominant** : sur les 50 fichiers sous-estimés, nombre moyen de clusters dont le majoritaire est le
   locuteur dominant : 1,28 → 2,96 ; il est coupé dans 45/50 fichiers. Les locuteurs « isolés » (majoritaires d'au moins un
   cluster) passent de 2,12 à 3,36 en moyenne pour 5,70 vrais locuteurs. Rappel pondéré par le temps : dominant 0,861 → 0,634 ;
   autres 0,518 → 0,718. Le gain sur les petits locuteurs est réel, mais il porte sur moins de temps que la perte sur le dominant.
3. **Corrélations** (Spearman, 50 fichiers sous-estimés) : ΔDER vs part du dominant ρ = +0,54 (p = 5e-5) ; vs n_ref
   ρ = −0,28 (p = 0,045) ; vs écart propre à n_ref ρ = +0,21 (p = 0,14, non significatif).
4. **Fusion oracle des clusters** (diagnostic : chaque cluster prend son locuteur majoritaire) : eigengap 25,32 %,
   k oracle **17,51 %**. La partition à k = n_ref contient donc plus d'information ; c'est le morcellement qui coûte.
5. **Balayage de k** (mêmes 79 fichiers avec n_ref ≥ 2) : eigengap 30,83 % ; k = n_ref 35,69 % ; k = n_ref − 1 36,98 % ;
   k = n_ref − 2 39,35 % ; meilleur k par fichier 22,77 %. Le meilleur k est n_ref dans 36/90 fichiers et < n_ref dans 51.
6. **JER** : il baisse avec le k oracle (61,13 → 53,46 %) alors que le DER monte.
7. **Embeddings** (VAD oracle, windows pures) : AUC même / différent locuteur médiane 0,789 (min 0,200, 78 fichiers) ;
   silhouette avec les labels de référence médiane 0,192. Ratio de Fisher médian plus élevé sur les moyennes de MFCC
   (d06 0,70, d08 0,62) que sur les écarts-types (≈ 0,11-0,14).

### Interprétation
- Le k oracle n'est « meilleur » que si l'espace des embeddings contient k groupes. Avec des MFCC 2 s et p = 10 %, les
  locuteurs de moins de p windows n'ont pas de bloc dans `A_bin`. Le spectre n'offre alors pas k valeurs propres proches de 0,
  et KMeans (`clustering.py:92`) découpe le grand bloc du dominant. Le DER apparie les clusters un à un : les morceaux en
  trop du dominant sont associés à un petit locuteur (ou à aucun), et tout leur temps devient de la confusion.
- L'eigengap est un estimateur **adaptatif** : il sous-estime k exactement là où les locuteurs en trop ne sont pas séparables.
  La preuve : un k uniformément réduit (n_ref − 1) fait pire que n_ref. L'eigengap ne gagne pas parce qu'il choisit « moins »,
  mais parce qu'il lit la structure réelle du graphe.
- Le DER favorise la sous-estimation : oublier un locuteur de 2 s coûte 2 s, couper un locuteur de 140 s en trois coûte
  jusqu'à 90 s. Le JER pondère chaque locuteur également, d'où le sens inverse.
- La VAD n'y est pour rien : même effet avec les 3 VAD, et presque aucun locuteur ne perd toutes ses windows. La limite est
  l'échelle des windows (2 s) et la taille du voisinage p, pas la VAD.

### Hypothèses non vérifiées
- Un choix de p adapté à chaque fichier (NME-SC) ou un nombre de voisins fixe réduirait le morcellement. Seuls p = 0.05 et
  0.02 ont été testés, et pas de façon monotone.
- Les fichiers à très faible AUC (min 0,200) sont probablement de la musique ou du bruit (non écouté).
- La seed du KMeans pourrait influer sur la façon dont le dominant est coupé (une seule seed testée).

## 5. Résultats surprenants
- Le k oracle produit une partition **plus pure** et plus « fusionnable » (17,51 % après fusion oracle), et pourtant un DER
  plus mauvais. C'est expliqué par le morcellement et l'appariement one-to-one (point 4).
- uvnmy : spk00 (2,2 s, 0 window majoritaire) et spk01 (1,8 s, 1 window) sont impossibles à isoler. Avec k = 4, deux morceaux
  de spk03 (49 s + 37 s) restent **sans appariement**, soit 86 s de confusion (figures/uvnmy_cas_k_oracle.png).
- La reproduction energy_vad diffère des CSV committés (33,95 % au lieu de 33,49 %) : `energy_vad` a changé au commit 7ade822,
  après ces runs. Avec la VAD oracle, la reproduction est exacte (28,67 %).

## 6. Pistes pour Grégoire (non implémentées)
1. **Comparabilité** : `baseline_oracle.py` appelle `spectral_clustering` sans p_neighbors ni max_speakers (défauts 0.1 / 10),
   alors que `baseline.py` passe 0.05 / 20. Les runs oracle actuels ne sont donc pas comparables à `baseline_sample90_seed130`
   (31,55 %). Deux paramètres ont aussi changé à la fois entre ces runs.
2. **Fuite** : `best_vad(..., train="alldev")` dans `baseline.py` et `baseline_oracle.py` évalue sur le sample 90 un modèle
   entraîné sur tout le dev. Utiliser `train="devtrain"` pour tout réglage sur le dev (30,14 % au lieu de 29,71 % ici).
3. Le « k oracle » n'est pas un bon plafond pour ce pipeline. Un plafond informatif est le meilleur k par fichier
   (22,77 % sur 79 fichiers), ou le nombre de locuteurs ayant au moins une window pure.
4. Pistes d'amélioration n°2 suggérées par le mécanisme : p_neighbors adaptatif (NME-SC), ou fusion a posteriori des clusters
   proches (ex. AHC sur les centroïdes). À tester avec un balayage, sur dev uniquement.

## 7. Fichiers sources inchangés
`git diff --stat -- embeddings.py clustering.py post_treatment.py baseline.py baseline_oracle.py audio_features.py vad.py`
donne exactement les modifications non committées présentes **avant** cette tâche : baseline.py 9, baseline_oracle.py 9,
clustering.py 3 lignes. Aucune modification de source, même temporaire, n'a été faite.

## 8. Reproduction
```
source .venv/bin/activate
python analysis/clustering_agent/eval_existing_rttm.py results/baseline_oracle_vad0_k0_vote0_sample90_seed130.rttm HEAD:results/baseline_oracle_vad0_k1_vote0_sample90_seed130.rttm
python analysis/clustering_agent/diag_k_oracle.py --tag sample90        # ~6,5 min
python analysis/clustering_agent/aggregate_k_oracle.py sample90
python analysis/clustering_agent/extra_stats_k_oracle.py
python analysis/clustering_agent/figures_k_oracle.py --cases uvnmy,bauzd,hkzpa
```
