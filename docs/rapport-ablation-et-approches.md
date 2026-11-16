# Comparaison des approches et analyse d'ablation

> Rédigé par Claude Code le 16/11/2026, à la demande de Grégoire. Détail de l'ablation : `docs/rapport-ablation.md`.
> Consultations du test pour ce document : 6e (baseline V3, run de Grégoire), 7e et 8e (baseline V2 et V2 + WeSpeaker,
> runs de Claude Code à la demande de Grégoire, versions fixées avant l'exécution, voir section 4).

Protocole commun (D-30 à D-37) : `pyannote.metrics`, collar 0, UEM = [0, durée du wav], agrégation micro. DER moderne
(`skip_overlap=False`) = métrique principale.

## 1. Tableau comparatif — test sample 90 (`samples/test/sample90_seed130.rttm`, 90 fichiers)

Toutes les lignes portent sur les **mêmes 90 fichiers** et le même protocole.

| Approche | DER moderne | DER forgiving | JER | FA | missed | confusion | k exact / sous-estimé / sur-estimé | RTF |
|---|---|---|---|---|---|---|---|---|
| Plancher « un seul locuteur » | 56,7 % | — | — | — | — | — | — | — |
| Baseline de départ : `energy_vad` + spectral p fixe 0.1 + stats MFCC | 44,31 % | 43,16 % | 77,19 % | 7,63 % | 8,06 % | 28,62 % | 17,8 / 73,3 / 8,9 % | 0,0012 (CPU) |
| + amélioration n°1 (`best_vad` `alldev`) | 36,89 % | 35,40 % | 77,00 % | 2,91 % | 6,39 % | 27,60 % | 13,3 / 77,8 / 8,9 % | 0,0042 (CPU) |
| + amélioration n°2 (NME-SC) | 38,38 % | 36,73 % | 68,47 % | 7,63 % | 8,06 % | 22,69 % | 16,7 / 64,4 / 18,9 % | 0,0041 (CPU) |
| Baseline améliorée (n°1 + n°2, ablation) | 30,86 % | 28,77 % | 68,38 % | 2,91 % | 6,39 % | 21,57 % | 18,9 / 71,1 / 10,0 % | 0,0079 (CPU) |
| **Baseline V2** : n°1 (`best_vad` `devtrain`) + n°2, stats MFCC | 30,77 % | 28,65 % | 67,32 % | 2,82 % | 6,88 % | 21,07 % | 16,7 / 72,2 / 11,1 % | 0,007 (CPU) |
| **Baseline V3** : V2 + `fill_unlabeled` | 30,07 % | 27,93 % | 67,04 % | 3,38 % | 5,18 % | 21,51 % | 16,7 / 72,2 / 11,1 % | 0,0081 (CPU) |
| **Baseline V2 + embeddings pré-entraînés** (WeSpeaker à la place des stats MFCC) | **17,14 %** | **14,26 %** | **43,36 %** | 2,82 % | 6,88 % | 7,44 % | 32,2 / 52,2 / 15,6 % | 0,019 (GPU, non représentatif) |
| **pyannote** `speaker-diarization-community-1` | **10,69 %** | **9,51 %** | **32,55 %** | n.d. | n.d. | n.d. | 42,2 / 43,3 / 14,4 % | 0,265 (GPU, non représentatif) |

Sources :
- ablation : `results/ablation/ablation_test_sample90_seed130_agregats.csv` ;
- V2 : `results/baselineV2 testset/` (sortie de `baseline.py`, `run.log`) ;
- V3 : `results/baselineV3 testset/` (agrégats recalculés depuis le RTTM avec le protocole complet ; écart max de DER par
  fichier avec le CSV : 8e-05) ;
- V2 + WeSpeaker : `results/baselineV2 pretrained embeddings testset/` (sortie de `baseline.py`, `run.log`) ;
- pyannote : `results/pyannote/pyannote_c1_sample90_seed130_par_fichier.csv` (run du 07/11).

Chaque ligne en gras ne diffère de la baseline V2 que par **un seul** changement : `fill_unlabeled` (V3) ou les
embeddings (V2 + WeSpeaker). La VAD de V2 / V3 / V2 + WeSpeaker est la même (`best_vad(train="devtrain")`, VAD error
5,76 %) ; FA et missed sont donc identiques entre V2 et V2 + WeSpeaker.

**Note sur la ligne pyannote** : son RTTM (`results/pyannote/pyannote_c1_sample90_seed130.rttm`) est vide (0 octet). Les
agrégats micro ont été **reconstruits** à partir du CSV par fichier : DER = Σ(DER_i × total_i) / Σ total_i (total_i =
parole de référence du fichier, calculée avec `DiarizationErrorRate` contre une hypothèse vide, même UEM) ; JER pondéré
par le nombre de locuteurs de référence. Contrôle : appliquée aux CSV de l'ablation, cette reconstruction redonne
**exactement** les agrégats officiels (44,31 / 43,16 / 77,19 % et 30,86 / 28,77 / 68,38 %). La décomposition FA / missed /
confusion n'est pas reconstructible sans RTTM (n.d.). Les RTF GPU sont mesurés sur un GPU bloqué en mode idle : ils ne
sont pas comparables à ceux de la baseline sur CPU.

**Recouvrement possible des données d'entraînement** (D-40) : `pyannote/segmentation-3.0` (dans community-1) a vu une
partie inconnue de VoxConverse ; WeSpeaker est entraîné sur VoxCeleb (non vérifié), qui peut partager des locuteurs avec
VoxConverse. Les lignes pyannote et V2 + WeSpeaker peuvent donc être optimistes.

### Écarts entre lignes voisines (DER moderne ; fichiers améliorés / dégradés de plus de 0,5 point)

| Passage (un seul changement) | Δ DER | Δ JER | améliorés / dégradés | fichiers avec un k différent |
|---|---|---|---|---|
| Baseline améliorée → V2 (VAD entraînée sur `devtrain` au lieu de `alldev`) | −0,09 | −1,06 | 20 / 23 | 17 |
| V2 → V3 (`fill_unlabeled`) | −0,70 | −0,28 | 36 / 4 | 0 |
| V2 → V2 + WeSpeaker (embeddings) | **−13,63** | **−23,96** | 72 / 7 | 61 |
| V2 + WeSpeaker → pyannote (pipeline complet) | −6,45 | −10,81 | 82 / 5 (en faveur de pyannote) | 50 |

- Changer les données d'entraînement de la VAD ne change pas le DER global (−0,09 point, autant de fichiers dégradés
  qu'améliorés) : c'est du bruit.
- `fill_unlabeled` : petit gain mais régulier (36 fichiers améliorés, 4 dégradés) ; il réduit le missed (6,88 → 5,18 %)
  au prix d'un peu de FA (2,82 → 3,38 %) et de confusion (21,07 → 21,51 %). k inchangé, comme attendu (post-traitement).
- WeSpeaker : seule la confusion change (21,07 → 7,44 %, divisée par 2,8). Les 10 fichiers les plus améliorés portent
  40 % du gain.

### Cas d'échec (test sample 90, DER moderne par fichier)

- **Baseline V2 / V3 (stats MFCC)** : 26 fichiers sur 90 au-dessus de 50 % de DER (V3). Pires cas : k = 1 sur des
  fichiers à nombreux locuteurs — qeejz (14 loc.) 86,1 %, pgtkk (9 loc.) 82,5 %, eddje (4 loc.) 71,2 % ; et euqef
  (1 loc., k = 4) 73,0 %. La sous-estimation de k touche 72 % des fichiers.
- **V2 + WeSpeaker** : 2 fichiers au-dessus de 50 % — vylyk (3 loc., k = 1) 58,8 %, jzkzt (11 loc., k = 6) 51,8 % ; puis
  wcxfk (3 loc., k = 4) 47,9 % et wprog (2 loc., k = 3) 46,9 %. k reste sous-estimé sur 52 % des fichiers.
- **pyannote** : 1 fichier au-dessus de 50 % — vylyk (3 loc., k = 1) 65,0 % ; puis fzwtp, jzkzt, vgevv (11 loc. chacun)
  entre 30,6 et 33,3 %. vylyk et jzkzt sont difficiles pour les deux approches pré-entraînées.
- DER médian par fichier selon le nombre de locuteurs de référence :

| Locuteurs de référence (fichiers) | V2 | V2 + WeSpeaker | pyannote |
|---|---|---|---|
| 1 (7) | 19,3 % | 5,6 % | 2,1 % |
| 2-3 (18) | 18,3 % | 16,2 % | 10,6 % |
| 4-6 (23) | 21,9 % | 16,9 % | 8,8 % |
| 7 et plus (42) | 43,8 % | 17,6 % | 10,5 % |

## 2. Comparaison des approches sur le dev (`samples/dev/sample90_seed130.rttm`, 90 fichiers)

**Non comparable au tableau 1** : autres fichiers, et réglages choisis sur ce même sample. VAD
`best_vad(train="devtrain")` (sans fuite), NME-SC, windows 2,0 s / 1,0 s, KMeans seed 42 dans les trois lignes. V3 et
V2 + WeSpeaker ne diffèrent de V2 que par un seul changement.

| Approche | DER moderne | DER forgiving | JER | k exact | RTF |
|---|---|---|---|---|---|
| **Baseline V2** : stats MFCC (référence) | 27,39 % | 24,48 % | 54,85 % | 43,3 % | 0,0065 (CPU) |
| **Baseline V3** : V2 + `fill_unlabeled` (trames de parole sans window → label voisin) | 26,53 % | 23,57 % | 54,47 % | 43,3 % | 0,0054 (CPU) |
| **V2 + WeSpeaker** (embeddings pré-entraînés à la place des stats MFCC) | 14,24 % | 11,02 % | 32,08 % | 55,6 % | 0,0186 (GPU, non représentatif) |

Sources (CSV par fichier, agrégats micro reconstruits comme en section 1) :
- V2 et V2 + WeSpeaker : `results/pretrained_emb/{mfcc_stats,wespeaker}_sample90_seed130_par_fichier.csv` (run du 15/11,
  analyse détaillée dans `results/pretrained_emb/rapport.md`). Ce run est **antérieur à `fill_unlabeled`** (absent du
  code jusqu'au commit `13682fc` du 16/11) : l'analyse WeSpeaker porte donc sur la **V2**. Contrôle : sa variante
  `mfcc_stats` est identique fichier par fichier à `results/baselineV2 devset/` (écart max de DER 0,0, mêmes k).
  Les agrégats redonnent ceux de `results/pretrained_emb/agregats_sample90_seed130.csv` ;
- V3 : `results/baselineV3 devset/baseline_sample90_seed130_par_fichier.csv`.

pyannote n'a pas été évalué sur ce sample du dev.

**Observé**
- WeSpeaker : −13,15 points de DER et −22,77 points de JER ; seule la confusion change (19,46 → 6,30 %), car la VAD et
  les windows sont identiques. Le gain est réparti sur beaucoup de fichiers : 55 améliorés, 6 dégradés à plus d'un
  point (`results/pretrained_emb/rapport.md`).
- `fill_unlabeled` : −0,86 point de DER (−0,70 sur le test, 36 fichiers améliorés / 4 dégradés). Gain faible mais
  de même signe sur le dev et le test.

**Interprétation — trois familles d'approches**
- **Baseline classique** (VAD apprise légère + stats MFCC + spectral) : la plus rapide (RTF < 0,01 sur CPU), mais les
  stats MFCC séparent mal les locuteurs, d'où une forte confusion et un k souvent sous-estimé.
- **Embeddings pré-entraînés dans notre pipeline** (WeSpeaker) : on garde notre VAD, nos windows et notre clustering, et
  on ne change que la représentation. La confusion est divisée par 3 sur le dev et par 2,8 sur le test (−13,63 points de
  DER). C'est le levier le plus fort mesuré dans le projet, et il se confirme sur le test.
- **Pipeline pré-entraîné complet** (pyannote community-1) : segmentation neuronale, gestion de l'overlap et embeddings
  appris. Sur le test, il reste devant la baseline V2 + WeSpeaker (10,69 % contre 17,14 %, 82 fichiers sur 90). L'écart
  restant porte surtout sur les fichiers à 2-6 locuteurs (section 1) ; FA / missed / confusion de pyannote ne sont pas
  disponibles pour dire d'où il vient.

## 3. Analyse d'ablation

Protocole fixé avant exécution (D-62) : test sample 90, 4 configurations croisant les deux améliorations, tout le reste
identique (frames, MFCC, windows 2 s / 1 s, `max_speakers` 20, KMeans seed 42, post-traitement).

**Observé**
- Amélioration n°1 (VAD `energy_vad` → `best_vad`) : DER −7,42 point (sans n°2) et −7,52 (avec n°2). Elle réduit la FA
  (7,63 → 2,91 %) et le missed (8,06 → 6,39 %), presque pas la confusion, et ne change pas le JER (77,19 → 77,00 %).
  74 fichiers améliorés, 11 dégradés.
- Amélioration n°2 (p fixe 0.1 → NME-SC) : DER −5,93 (sans n°1) et −6,03 (avec n°1). Elle réduit uniquement la
  confusion (28,62 → 22,69 %) et le JER (77,19 → 68,47 %). 39 fichiers améliorés, 18 dégradés.
- Ensemble : 44,31 → 30,86 % (−13,45 points) ; interaction −0,10 point.

**Interprétation**
- Les gains s'additionnent car les deux améliorations corrigent des erreurs différentes : détection (n°1) et attribution
  des locuteurs (n°2). L'interaction de 0,1 point est du niveau du bruit.
- La n°1 est plus régulière fichier par fichier que la n°2.
- Après les deux améliorations, la confusion reste la première source d'erreur (21,57 sur 30,86 points), surtout par
  sous-estimation de k sur les fichiers à nombreux locuteurs. Les embeddings pré-entraînés (V2 + WeSpeaker, 17,14 %) et
  pyannote (10,69 %) corrigent surtout ce point.

**Limites / hypothèses**
- Une seule seed KMeans et un seul sample de 90 fichiers : pas d'intervalle de confiance.
- Le gain de la n°2 dépend de la référence choisie (p = 0.1 ; sur le dev contre p = 0.02, DER −1,7 point mais JER dégradé).
- Le DER de V2 + WeSpeaker peut être optimiste si VoxCeleb partage des locuteurs avec VoxConverse (non vérifié).

## 4. Reproduction

```bash
source .venv/bin/activate
python ablation.py       # 4 configurations de l'ablation, test sample 90
python baseline.py       # V3 : sample_rttm / audio_dir sur le test, pretrained_embeddings = False,
                         # fill_unlabeled = True, out_dir = results/baselineV3 testset/
```

V2 et V2 + WeSpeaker : `baseline.py` actuel, sans modification du fichier, avec seulement ces paramètres changés
(lanceur qui importe le module et appelle `main()`) : `sample_rttm = "samples/test/sample90_seed130.rttm"`,
`audio_dir = data/voxconverse/test/voxconverse_test_wav`, `fill_unlabeled = False`, `pretrained_embeddings = False`
(V2) ou `True` (V2 + WeSpeaker), `out_dir` = `results/baselineV2 testset/` ou
`results/baselineV2 pretrained embeddings testset/`. VAD `devtrain` (défaut de `diarize_file`), NME-SC, KMeans seed 42.
Équivalent : mettre ces valeurs en tête de `baseline.py` puis `python baseline.py`.
