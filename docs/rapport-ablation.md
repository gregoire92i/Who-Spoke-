# Rapport d'ablation — 2 améliorations de la baseline

> Rédigé par Claude Code le 14/11/2026, à la demande de Grégoire. Protocole défini par Grégoire et noté dans
> `docs/choix_techniques.md` (D-62, révision du 14/11/2026) **avant** l'exécution. Résultats recopiés de
> `results/ablation/ablation_test_sample90_seed130_agregats.csv` et des CSV par fichier du même dossier.

## 1. Objectif

Mesurer l'apport de chacune des 2 améliorations retenues pour le rendu, seule et combinée à l'autre :

- **Amélioration n°1 — VAD** : `energy_vad` de base (seuil −30 dB, sans hystérésis ni lissage de l'énergie) →
  `best_vad` (VAD apprise `vad_stack`, D-57, D-62).
- **Amélioration n°2 — clustering** : nombre de voisins du graphe spectral **fixe** (`p_neighbors = 0.1`) → **choisi
  fichier par fichier** par NME-SC (`nme_spectral_clustering`, Laplacien normalisé, D-59).

## 2. Protocole

| Élément | Valeur |
|---|---|
| Fichiers | **test sample 90** (`samples/test/sample90_seed130.rttm`, audio `data/voxconverse/test/voxconverse_test_wav/`), 90 fichiers, aucun manquant |
| Consultation du test | **5e consultation** (les 4 premières par le subagent `vad-tester`). Version figée dans `ablation.py` avant l'exécution ; aucun passage préalable sur le dev ; aucun paramètre choisi d'après ce résultat |
| VAD avant / après | `energy_vad(seuil_db=-30, delta_db=0, smooth_frames=1, min_silence=0.5, min_speech=0.25)` / `best_vad(train="alldev")` (modèles entraînés sur les 216 fichiers dev uniquement) |
| Clustering avant / après | `spectral_clustering(p_neighbors=0.1, min_neighbors=5)` / `nme_spectral_clustering(laplacian="normalized", 30 valeurs de p entre 5 voisins et 25 % des windows)` ; dans les deux cas `min_speakers = 1`, `max_speakers = 20`, embeddings centrés-réduits, KMeans seed 42 |
| Identique partout | 16 kHz mono ; frames 65 ms / hop 32,5 ms ; 13 MFCC ; windows 2,0 s / hop 1,0 s / min 0,5 s (moyenne + écart-type, mode `segment`) ; post-traitement `windows_to_annotation` |
| Métriques (D-30 à D-37) | collar 0 ; UEM = [0, durée du wav] ; agrégation micro ; DER moderne (principal), DER forgiving, JER, FA / missed / confusion ; detection error de la VAD ; RTF |

Pour chaque fichier, l'audio, la STFT et les MFCC sont calculés une seule fois ; chaque VAD produit ses embeddings,
sur lesquels les deux clusterings sont appliqués. Deux configurations voisines ne diffèrent donc que par une amélioration.

## 3. Résultats observés

### 3.1 Agrégats micro (test sample 90)

| Configuration | DER moderne | DER forgiving | JER | FA | missed | confusion | detection error VAD | RTF |
|---|---|---|---|---|---|---|---|---|
| 1. `energy_vad` + p fixe 0.1 (**départ**) | 44,31 % | 43,16 % | 77,19 % | 7,63 % | 8,06 % | 28,62 % | 10,94 % | ≤ 0,01 |
| 2. `best_vad` + p fixe 0.1 (+ n°1) | 36,89 % | 35,40 % | 77,00 % | 2,91 % | 6,39 % | 27,60 % | 5,43 % | ≤ 0,01 |
| 3. `energy_vad` + NME-SC (+ n°2) | 38,38 % | 36,73 % | 68,47 % | 7,63 % | 8,06 % | 22,69 % | 10,94 % | ≤ 0,01 |
| 4. `best_vad` + NME-SC (**les deux**) | **30,86 %** | **28,77 %** | **68,38 %** | 2,91 % | 6,39 % | 21,57 % | 5,43 % | ≤ 0,01 |

Référence : plancher « un seul locuteur » sur ce sample = 56,7 % de DER moderne.

### 3.2 Gain de chaque amélioration (DER moderne, en points)

| | Amélioration n°1 (VAD) | Amélioration n°2 (NME-SC) |
|---|---|---|
| L'autre amélioration absente | −7,42 (1 → 2) | −5,93 (1 → 3) |
| L'autre amélioration présente | −7,52 (3 → 4) | −6,03 (2 → 4) |

- Gain total (1 → 4) : **−13,45 points** de DER moderne, −14,39 points de DER forgiving, −8,81 points de JER.
- Terme d'interaction : −7,52 − (−7,42) = −0,10 point.

### 3.3 Décomposition des erreurs

| Passage | Δ FA | Δ missed | Δ confusion |
|---|---|---|---|
| n°1 (1 → 2) | −2 796 s | −991 s | −603 s |
| n°1 (3 → 4) | −2 796 s | −991 s | −665 s |
| n°2 (1 → 3) | 0 s | 0 s | −3 510 s |
| n°2 (2 → 4) | 0 s | 0 s | −3 572 s |
| les deux (1 → 4) | −2 796 s | −991 s | −4 175 s |

### 3.4 Estimation du nombre de locuteurs k

| Configuration | k exact | k sous-estimé | k surestimé |
|---|---|---|---|
| 1. `energy_vad` + p fixe | 17,78 % | 73,33 % | 8,89 % |
| 2. `best_vad` + p fixe | 13,33 % | 77,78 % | 8,89 % |
| 3. `energy_vad` + NME-SC | 16,67 % | 64,44 % | 18,89 % |
| 4. `best_vad` + NME-SC | 18,89 % | 71,11 % | 10,00 % |

### 3.5 Fichier par fichier (DER moderne, seuil de 0,5 point)

| Passage | fichiers améliorés | fichiers dégradés | fichiers avec un k différent |
|---|---|---|---|
| n°1 (1 → 2) | 74 | 11 | 38 |
| n°1 (3 → 4) | 70 | 16 | 48 |
| n°2 (1 → 3) | 39 | 18 | 43 |
| n°2 (2 → 4) | 33 | 16 | 37 |
| les deux (1 → 4) | 74 | 13 | 48 |

Cas extrêmes (DER avant → après) :

- n°1 (1 → 2) : améliorés euqef 131,7 → 68,1 %, wprog 86,8 → 36,8 %, vylyk 114,7 → 70,2 % ; dégradés isrps
  37,4 → 77,0 %, bgvvt 25,8 → 45,9 %, lpola 40,2 → 54,3 %.
- n°2 (2 → 4) : améliorés isrps 77,0 → 24,7 %, ezxso 75,5 → 29,9 %, swbnm 60,6 → 16,3 % ; dégradés zedtj
  30,4 → 59,4 %, mclsr 31,1 → 54,3 %, eddje 51,5 → 71,0 %.
- les deux (1 → 4) : améliorés kvkje 92,3 → 18,7 %, euqef 131,7 → 68,1 %, wprog 86,8 → 38,9 % ; dégradés diysk
  68,1 → 81,3 %, lpola 40,2 → 54,3 %, mclsr 32,1 → 54,3 %.

## 4. Interprétation

- **Les deux améliorations sont efficaces et indépendantes.** Chacune réduit le DER moderne de 6 à 7,5 points, que
  l'autre soit présente ou non ; l'interaction (−0,10 point) est négligeable. C'est attendu : elles corrigent des erreurs
  de nature différente.
- **L'amélioration n°1 agit sur la détection de parole** : la FA chute de 7,63 % à 2,91 % et le missed de 8,06 % à
  6,39 %. Son effet sur la confusion est faible (−603 s), et elle ne change pas le JER (77,19 → 77,00 %). Elle est
  robuste fichier par fichier (74 améliorés contre 11 dégradés).
- **L'amélioration n°2 agit uniquement sur la confusion** (28,62 % → 22,69 %), à VAD identique. Elle réduit la
  sous-estimation de k (73 % → 64 % des fichiers avec `energy_vad`), ce qui améliore aussi fortement le JER
  (77,19 → 68,47 %). Elle est moins régulière fichier par fichier (39 améliorés contre 18 dégradés).
- **Lien avec le diagnostic du k oracle** (`results/diag_k_oracle_vs_eigengap/rapport.md`) : avec p = 10 % des windows,
  les locuteurs minoritaires n'ont pas de groupe propre dans le graphe et l'eigengap sous-estime k ; NME-SC choisit un p
  adapté à chaque fichier, ce qui leur laisse plus souvent un groupe.
- **Différence avec l'expérience sur le dev** (`results/nme_sc/`) : sur le dev, NME-SC était comparé à p = 0.02, qui
  surestimait souvent k ; il y réduisait le DER de 1,7 point mais dégradait le JER. Ici, la référence p = 0.1 sous-estime
  k : NME-SC améliore alors les deux métriques. Les deux expériences ne sont pas comparables directement (référence,
  VAD et fichiers différents).

## 5. Hypothèses (non vérifiées)

- isrps est le fichier le plus dégradé par l'amélioration n°1 (37,4 → 77,0 %) et le plus amélioré par la n°2
  (77,0 → 24,7 %) : la nouvelle VAD y provoquerait une mauvaise estimation de k, que NME-SC corrige.
- Les fichiers les plus dégradés au total (diysk, lpola, mclsr) n'ont pas été écoutés ; diysk figurait déjà parmi les
  fichiers à fausses alarmes de la VAD (musique ou foule, selon le rapport de `vad-tester`).
- La confusion reste la première source d'erreur après les deux améliorations (21,57 % sur 30,86 %) : des embeddings
  plus discriminants que les statistiques de MFCC seraient le levier suivant.

## 6. Limites

- **Une seule seed de KMeans** (42) : la variabilité des gains n'est pas mesurée. Les écarts de l'ordre de 0,1 point
  (interaction) relèvent du bruit.
- **Un seul sample de 90 fichiers** du test : pas d'intervalle de confiance.
- **RTF** : ≤ 0,01 pour toutes les configurations sur ce PC (CPU) ; le temps de `best_vad` inclut le calcul de ses
  propres caractéristiques.
- Le choix de `p_neighbors = 0.1` comme référence a été fait par Grégoire avant l'exécution ; avec une autre référence
  (par exemple 0.02), le gain de l'amélioration n°2 serait différent (voir l'expérience sur le dev).
- Pas de comparaison avec pyannote dans ce rapport (tableau comparatif à faire séparément, mêmes fichiers).

## 7. Reproduction et fichiers

```bash
source .venv/bin/activate
python ablation.py          # ~15-20 min, 90 fichiers du test sample 90
```

- Script : `ablation.py` (paramètres figés en tête de fichier).
- Résultats : `results/ablation/`
  - `ablation_test_sample90_seed130_agregats.csv` ;
  - `ablation_test_sample90_seed130_{energy_fixe,best_fixe,energy_nme,best_nme}_par_fichier.csv` ;
  - `ablation_test_sample90_seed130_{energy_fixe,best_fixe,energy_nme,best_nme}.rttm` ;
  - `run_test_sample90_seed130.log`.
- Code des améliorations : `vad.py` (`best_vad`), `clustering.py` (`nme_spectral_clustering`).
