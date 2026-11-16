# Comparaison des approches et analyse d'ablation

> Rédigé par Claude Code le 16/11/2026, à la demande de Grégoire, à partir des résultats déjà présents dans le dépôt
> (aucune nouvelle exécution sur le test). Détail de l'ablation : `docs/rapport-ablation.md`.
> **Ligne « baseline finale » sur le test : en attente du run de Grégoire** (6e consultation du test).

Protocole commun (D-30 à D-37) : `pyannote.metrics`, collar 0, UEM = [0, durée du wav], agrégation micro. DER moderne
(`skip_overlap=False`) = métrique principale.

## 1. Tableau comparatif — test sample 90 (`samples/test/sample90_seed130.rttm`, 90 fichiers)

Toutes les lignes portent sur les **mêmes 90 fichiers** et le même protocole.

| Approche | DER moderne | DER forgiving | JER | FA | missed | confusion | k exact / sous-estimé / sur-estimé | RTF |
|---|---|---|---|---|---|---|---|---|
| Plancher « un seul locuteur » | 56,7 % | — | — | — | — | — | — | — |
| Baseline de départ : `energy_vad` + spectral p fixe 0.1 + stats MFCC | 44,31 % | 43,16 % | 77,19 % | 7,63 % | 8,06 % | 28,62 % | 17,8 / 73,3 / 8,9 % | 0,0012 (CPU) |
| + amélioration n°1 (`best_vad`) | 36,89 % | 35,40 % | 77,00 % | 2,91 % | 6,39 % | 27,60 % | 13,3 / 77,8 / 8,9 % | 0,0042 (CPU) |
| + amélioration n°2 (NME-SC) | 38,38 % | 36,73 % | 68,47 % | 7,63 % | 8,06 % | 22,69 % | 16,7 / 64,4 / 18,9 % | 0,0041 (CPU) |
| Baseline améliorée (n°1 + n°2) | **30,86 %** | **28,77 %** | **68,38 %** | 2,91 % | 6,39 % | 21,57 % | 18,9 / 71,1 / 10,0 % | 0,0079 (CPU) |
| Baseline finale (`baseline.py` actuel : + `fill_unlabeled` + WeSpeaker) | *à mesurer* | *à mesurer* | *à mesurer* | | | | | |
| pyannote `speaker-diarization-community-1` | **10,69 %** | **9,51 %** | **32,55 %** | n.d. | n.d. | n.d. | 42,2 / 43,3 / 14,4 % | 0,265 (GPU, non représentatif) |

Sources : `results/ablation/ablation_test_sample90_seed130_agregats.csv` ;
`results/pyannote/pyannote_c1_sample90_seed130_par_fichier.csv` (run du 07/11).

**Note sur la ligne pyannote** : son RTTM (`results/pyannote/pyannote_c1_sample90_seed130.rttm`) est vide (0 octet). Les
agrégats micro ont été **reconstruits** à partir du CSV par fichier : DER = Σ(DER_i × total_i) / Σ total_i (total_i =
parole de référence du fichier, calculée avec `DiarizationErrorRate` contre une hypothèse vide, même UEM) ; JER pondéré
par le nombre de locuteurs de référence. Contrôle : appliquée aux CSV de l'ablation, cette reconstruction redonne
**exactement** les agrégats officiels (44,31 / 43,16 / 77,19 % et 30,86 / 28,77 / 68,38 %). La décomposition FA / missed /
confusion n'est pas reconstructible sans RTTM (n.d.). Le RTF de pyannote est mesuré sur un GPU bloqué en mode idle : il
n'est pas comparable à celui de la baseline (CPU).

**Recouvrement possible des données d'entraînement** (D-40) : `pyannote/segmentation-3.0` (dans community-1) a vu une
partie inconnue de VoxConverse ; WeSpeaker est entraîné sur VoxCeleb (non vérifié), qui peut partager des locuteurs avec
VoxConverse. Les lignes pyannote et « baseline finale » peuvent donc être optimistes.

### Cas d'échec (test sample 90, DER moderne par fichier)

- **Baseline améliorée** : 25 fichiers sur 90 au-dessus de 50 % de DER. Pire cas : k = 1 sur des fichiers à nombreux
  locuteurs — qeejz (14 loc.) 86,5 %, pgtkk (9 loc.) 82,3 %, diysk (15 loc.) 81,3 %. La sous-estimation de k touche
  71 % des fichiers.
- **pyannote** : 1 fichier au-dessus de 50 % — vylyk (3 loc., k = 1) 65,0 % ; puis fzwtp, jzkzt, vgevv (11 loc. chacun)
  entre 30,6 et 33,3 %.
- pyannote fait mieux que la baseline améliorée sur 88 fichiers sur 90 (écart > 0,5 point), moins bien sur 1.
- DER médian par fichier selon le nombre de locuteurs de référence (baseline améliorée → pyannote) : 1 loc. (7 fichiers)
  18,5 → 2,1 % ; 2-3 loc. (18) 18,4 → 10,6 % ; 4-6 loc. (23) 22,6 → 8,8 % ; 7 et plus (42) 45,2 → 10,5 %.

## 2. Comparaison des approches sur le dev (`samples/dev/sample90_seed130.rttm`, 90 fichiers)

**Non comparable au tableau 1** : autres fichiers, et réglages choisis sur ce même sample. VAD
`best_vad(train="devtrain")` (sans fuite), NME-SC, windows 2,0 s / 1,0 s, KMeans seed 42 dans les quatre lignes. Une
ligne ne diffère de sa voisine que par un seul changement.

| Approche | DER moderne | DER forgiving | JER | k exact | RTF |
|---|---|---|---|---|---|
| Baseline améliorée : stats MFCC (référence) | 27,39 % | 24,48 % | 54,85 % | 43,3 % | 0,0065 (CPU) |
| + `fill_unlabeled` (trames de parole sans window → label voisin) | 26,53 % | 23,57 % | 54,47 % | 43,3 % | 0,0054 (CPU) |
| Embeddings pré-entraînés WeSpeaker à la place des stats MFCC | 14,24 % | 11,02 % | 32,08 % | 55,6 % | 0,0186 (GPU, non représentatif) |
| WeSpeaker + `fill_unlabeled` (= `baseline.py` actuel) | **13,20 %** | **9,93 %** | **31,43 %** | 55,6 % | 0,0191 (GPU probable, non vérifié) |

Sources (CSV par fichier, agrégats micro reconstruits comme en section 1) :
`results/pretrained_emb/{mfcc_stats,wespeaker}_sample90_seed130_par_fichier.csv`,
`results/baselineV3 devset/baseline_sample90_seed130_par_fichier.csv`, `results/baseline_sample90_seed130_par_fichier.csv`.
Les deux premières lignes redonnent les agrégats officiels de `results/pretrained_emb/agregats_sample90_seed130.csv`.
pyannote n'a pas été évalué sur ce sample du dev.

**Observé**
- WeSpeaker : −13,15 points de DER et −22,77 points de JER ; seule la confusion change (19,46 → 6,30 %), car la VAD et
  les windows sont identiques. Le gain est réparti sur beaucoup de fichiers : 55 améliorés, 6 dégradés à plus d'un
  point (`results/pretrained_emb/rapport.md`).
- `fill_unlabeled` : −0,86 point de DER avec les MFCC, −1,04 avec WeSpeaker. Le gain est faible mais il apparaît avec les
  deux types d'embeddings.

**Interprétation — trois familles d'approches**
- **Baseline classique** (VAD apprise légère + stats MFCC + spectral) : la plus rapide (RTF < 0,01 sur CPU), mais les
  stats MFCC séparent mal les locuteurs, d'où une forte confusion et un k souvent sous-estimé.
- **Embeddings pré-entraînés dans notre pipeline** (WeSpeaker) : on garde notre VAD, nos windows et notre clustering, et
  on ne change que la représentation. La confusion est divisée par 3. C'est le levier le plus fort mesuré dans le projet.
- **Pipeline pré-entraîné complet** (pyannote community-1) : segmentation neuronale, gestion de l'overlap et embeddings
  appris. Sur le test, il reste loin devant la baseline améliorée (10,69 % contre 30,86 %). Sa comparaison avec la
  baseline WeSpeaker reste à faire sur le test (ligne « baseline finale »).

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
  sous-estimation de k sur les fichiers à nombreux locuteurs. C'est là que pyannote fait la différence (10,69 % de DER).

**Limites / hypothèses**
- Une seule seed KMeans et un seul sample de 90 fichiers : pas d'intervalle de confiance.
- Le gain de la n°2 dépend de la référence choisie (p = 0.1 ; sur le dev contre p = 0.02, DER −1,7 point mais JER dégradé).
- La confusion restante pointe vers les embeddings : sur le dev, WeSpeaker la divise par 3 (section 2). Effet sur le
  test à confirmer par la ligne « baseline finale ».

## 4. Reproduction

```bash
source .venv/bin/activate
python ablation.py       # 4 configurations de l'ablation, test sample 90
python baseline.py       # baseline finale (après passage sur samples/test/, voir en-tête du fichier)
```
