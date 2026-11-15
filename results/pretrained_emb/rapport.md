# Embeddings pré-entraînés WeSpeaker vs statistiques de MFCC (dev sample 90)

> Run du 15/11/2026, Claude Code, à la demande de Grégoire. Dev uniquement, test **non consulté**.
> Documents lus : `CLAUDE.md`, `docs/etat_du_projet.md`, `docs/choix_techniques.md` (D-40, D-56 à D-59, D-73, D-77).
> **D-41 n'existe pas** dans `docs/choix_techniques.md` (ni dans la copie de travail, ni dans HEAD `8571c1e`) :
> la section 5 ne contient que D-40. Le risque de fuite est traité ci-dessous en s'appuyant sur D-40.

## 1. Protocole

**Ce qui change (et seulement ça)** : le vecteur de chaque window.
- `mfcc_stats` : moyenne et écart-type des 13 MFCC (26 dimensions), comportement actuel (`pretrained_embeddings = False`).
- `wespeaker` : `pyannote/wespeaker-voxceleb-resnet34-LM` (256 dimensions), appliqué à l'audio de chaque window
  [start, end[ (`pretrained_embeddings = True`).

**Ce qui reste identique** (paramètres lus dans `baseline.py`, les deux variantes passent par `baseline.diarize_file`) :
- fichiers : `samples/dev/sample90_seed130.rttm` (90 fichiers, 27 666 s de parole de référence) ;
- VAD `best_vad(train="devtrain")`, entraînée hors sample 90 : pas de fuite VAD ;
- bornes des windows : `win_segs` de `compute_embeddings` (win 2,0 s, hop 1,0 s, min 0,5 s, mode `segment`) ;
- clustering NME-SC (Laplacien normalisé, `nme_max_frac` 0,25, `nme_n_grid` 30, `min_neighbors` 5),
  k par eigengap entre 1 et 20, `standardize = True`, seed 42 ;
- `windows_to_annotation` (vote des windows → frames ; le départage utilise le X de la variante) ;
- métriques : collar 0, UEM [0, durée du wav], agrégation micro (D-30 à D-36).

Le modèle est chargé une seule fois par processus (`embeddings.load_pretrained_model`), avec `token=False` : il n'est
pas soumis à conditions, **aucun jeton n'a été demandé ni utilisé**. Il tourne sur le GPU s'il y en a un. Le
chargement et une passe de chauffe (5,8 s) ne sont pas comptés dans le RTF.

## 2. Vérifications

| Vérification | Résultat observé |
|---|---|
| Mêmes bornes de windows dans les deux variantes | **90/90 fichiers** (`same_bounds` = True ; demandé sur 2 fichiers, vérifié sur tous) |
| `pretrained_embeddings = False` redonne la référence | DER 27,392 % ; écart max de DER par fichier avec `results/nme_sc/nme_normalized_sample90_seed130_par_fichier.csv` = 0,0, k identiques sur les 90 fichiers |
| NaN, pqmho (dev, hors sample 90, `--nan_check pqmho`) | 10 windows (2,01 et 2,02 s), k_hyp 1, aucun NaN. La VAD devtrain n'y produit **aucune** window courte |
| NaN, windows de durée minimale | 26 windows de 0,4875 s (`min_s` = 0,5 s arrondi à 15 frames de 32,5 ms) sur pqmho : aucun NaN ; 3 windows d'audio nul : aucun NaN |
| Windows courtes dans le run | 191 windows < 2 s sur 24 935 (43 fichiers), sans NaN (garde-fou `np.isfinite` dans `compute_pretrained_embeddings`) |
| `tests/test_demo.py` | **16 passed** |

## 3. Résultats observés (dev sample 90, micro)

| Variante | DER moderne | DER forgiving | JER | FA | Missed | Confusion | RTF | Appareil des embeddings |
|---|---|---|---|---|---|---|---|---|
| `mfcc_stats` (baseline) | 27,39 % | 24,48 % | 54,85 % | 1,62 % | 6,32 % | 19,46 % | 0,007 | CPU |
| `wespeaker` | **14,24 %** | **11,02 %** | **32,08 %** | 1,62 % | 6,32 % | **6,30 %** | 0,019 | GPU (RTX 3050 Laptop) |

- VAD error (identique pour les deux) : 3,131 %. FA et missed sont identiques fichier par fichier : seule la
  confusion change (5 383 s → 1 743 s).
- RTF : la variante `wespeaker` comprend VAD + MFCC (nécessaires pour les `win_segs`) + WeSpeaker + clustering.
  Par fichier : médiane 0,0183 (min 0,0098, max 0,0331) contre 0,0047 pour `mfcc_stats`. RTF GPU **non représentatif**
  sur ce PC (GPU bloqué en mode idle, CLAUDE.md §4). Le RTF de WeSpeaker sur CPU n'a pas été mesuré.

### Fichiers améliorés / dégradés (DER moderne par fichier)

| Seuil d'écart | Améliorés | Dégradés |
|---|---|---|
| > 0 point | 63 | 11 (16 identiques) |
| > 1 point | 55 | 6 |
| > 5 points | 43 | 3 |

JER : 64 améliorés, 11 dégradés, 15 identiques.

Plus gros gains (en secondes de confusion) : hgdez (10 loc., k 1 → 10, DER 76,0 → 14,7 %), oklol (7 loc., k 3 → 6,
52,0 → 11,2 %), kdfqk (20 loc., k 10 → 7, 74,6 → 34,2 %), ldnro (15 loc., k 9 → 16, 26,1 → 7,1 %), xxwgv (10 loc.,
k 4 → 9, 30,9 → 7,5 %). Les 10 meilleurs fichiers représentent **62,1 %** du gain total en confusion (3 640 s).

Dégradations de plus de 1 point : oxxwk (2 loc., k 2 → 1, DER 6,6 → 27,6 %), femmv (4 loc., k 3 → 5, 11,1 → 23,9 %),
zfkap (4 loc., k 3 → 2, 14,1 → 20,4 %), iwdjy (3 loc., k 1 → 2, 27,3 → 31,5 %), bkwns (2 loc., k 1 → 2, 22,2 → 26,2 %),
uvnmy (4 loc., k 2 → 2, 5,3 → 7,1 %).

### Erreurs sur k (eigengap)

| Variante | k exact | k sous-estimé | k sur-estimé | Erreur absolue moyenne |
|---|---|---|---|---|
| `mfcc_stats` | 39 | 42 | 9 | 1,556 |
| `wespeaker` | 50 | 33 | 7 | 0,733 |

k plus proche de la référence avec WeSpeaker sur 34 fichiers, plus éloigné sur 9. Sur les 42 fichiers sous-estimés
avec MFCC, 26 restent sous-estimés avec WeSpeaker.

Par nombre de locuteurs de référence (DER moderne micro, MFCC → WeSpeaker) : 1 loc. (11 fichiers) 7,59 → 6,95 % ;
2-3 loc. (38) 16,33 → 12,00 % ; 4-6 loc. (25) 33,09 → 20,05 % ; 7 et plus (16) 40,97 → 14,30 %.

## 4. Risque de fuite VoxCeleb / VoxConverse (non vérifié)

- D'après son nom, le modèle a été entraîné sur VoxCeleb. **Corpus et split exacts non vérifiés** : la carte du modèle
  n'est pas dans le cache local (seul `pytorch_model.bin` a été téléchargé) et n'a pas été lue.
- VoxCeleb et VoxConverse proviennent tous deux de vidéos YouTube (interviews, débats, journaux télévisés). Un
  recouvrement de **locuteurs** (personnalités publiques) ou de **vidéos** est possible ; il **n'a pas été vérifié**.
- Le D-41 cité dans la consigne est absent du dépôt. Selon D-40 (règle 3), ce recouvrement possible doit être signalé
  dans le tableau de résultats. Une mesure sur le test ne supprimerait pas ce risque (il concerne les locuteurs, pas le
  split VoxConverse).

## 5. Interprétation

- Le gain (−13,15 points de DER moderne, −22,77 points de JER) vient **entièrement de la confusion**, ce qui est
  attendu puisque la VAD et les windows sont identiques. Il est grand devant la variabilité habituelle et concerne une
  large majorité de fichiers (55 améliorés contre 6 dégradés à plus d'un point) : il est **peu probable que ce soit du
  bruit**. Il reste cependant concentré : 10 fichiers longs à nombreux locuteurs portent 62 % du gain.
- DER et JER s'améliorent ensemble, contrairement à NME-SC (D-59) où ils divergeaient.
- Les embeddings WeSpeaker semblent mieux séparer les locuteurs dans le graphe : l'eigengap trouve plus souvent le
  bon k, surtout pour les fichiers à 7 locuteurs ou plus. La sous-estimation de k reste l'erreur dominante
  (33 fichiers).
- Les réglages de la baseline (windows de 2 s, NME-SC, `standardize`) ont été choisis sur ce même sample avec les
  MFCC : la comparaison favorise plutôt les MFCC que WeSpeaker.
- Ce chiffre peut être optimiste si VoxCeleb recouvre des locuteurs de VoxConverse (section 4).

## 6. Hypothèses (non testées, aucun réglage fait pour favoriser WeSpeaker)

1. `standardize = True` (centrage-réduction par fichier avant le cosinus) a été réglé pour les MFCC ; pour WeSpeaker,
   le cosinus sur les embeddings bruts est l'usage courant. Il pourrait faire mieux ou moins bien.
2. Les windows de 2,0 s / hop 1,0 s (D-56) ont été réglées pour les MFCC ; WeSpeaker pourrait préférer d'autres
   longueurs (le modèle est spécifié pour des chunks de 5 s).
3. La grille NME-SC (`nme_max_frac`, `nme_n_grid`) et le Laplacien (normalisé ou non) ont été réglés avec les MFCC.
4. Les dégradations (oxxwk, zfkap : k sous-estimé ; femmv : k sur-estimé) pourraient venir de l'eigengap plutôt que
   des embeddings (à confirmer avec le mode k oracle).
5. Les embeddings du pipeline `speaker-diarization-community-1` (déjà en cache) sont peut-être le même ResNet34 : non
   vérifié.

Tout balayage de ces paramètres devrait se faire sur le dev, un seul changement à la fois.

## 7. Fichiers et reproduction

- Créés : `pretrained_embeddings_experiment.py`, `results/pretrained_emb/` (ce rapport,
  `agregats_sample90_seed130.csv`, `{mfcc_stats,wespeaker}_sample90_seed130_par_fichier.csv`,
  `{mfcc_stats,wespeaker}_sample90_seed130.rttm`, `run_sample90_seed130.log`).
- Modifiés : `embeddings.py` (ajout en fin de fichier de `load_pretrained_model`, `compute_pretrained_embeddings`,
  `WESPEAKER_ID`), `baseline.py` (paramètre `pretrained_embeddings = False`, import, branchement dans `diarize_file`).
- Nouvelle dépendance : aucune (pyannote.audio 4.0.7 et torch déjà installés). Modèle téléchargé
  dans `~/.cache/huggingface/hub/`.

```bash
source .venv/bin/activate
python pretrained_embeddings_experiment.py                 # dev sample 90, les deux variantes
python pretrained_embeddings_experiment.py --nan_check pqmho
python -m pytest tests/test_demo.py -q                     # 16 passed
```
