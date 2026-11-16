# État du projet — Who Spoke?

> Dernière mise à jour : 16/11/2026 (par le subagent `project-verifier`, à la demande de Grégoire). Mis à jour par
> Grégoire, ou par Claude Code uniquement après validation de Grégoire. Les décisions détaillées sont dans
> `docs/choix_techniques.md`. Dernier rapport de synchro : `docs/sync/2026-11-15_sync.md`.

## Calendrier

- Début : 26/10/2026. Durée : 3 semaines.
- Semaine 1 : corpus et baseline classique (terminée). Semaine 2 : pyannote (fait). Semaine 3 : améliorations
  mesurées, ablation, tableau comparatif, démo et documentation (faits au 16/11) ; soutenance à venir.

## Pipeline actuel

| Étape | Baseline (`baseline.py`, V3) | Variante / alternative |
|---|---|---|
| VAD | `best_vad` / `vad_stack` (`vad.py`, VAD apprise, D-57 et D-62), modèles `devtrain` par défaut (`diarize_file(..., vad_train="devtrain")`) | `energy_vad` sans hystérésis (`audio_features.py`, référence de l'amélioration n°1, D-53, D-54) ; `best_vad(train="alldev")` dans la démo et l'ablation ; VAD oracle (`baseline_oracle.py`, pas à jour) |
| Caractéristiques | STFT (frames 65 ms, hop 32,5 ms) + 13 MFCC (`audio_features.py`) | — |
| Embeddings | statistiques des MFCC (moyenne, écart-type) par windows de 2,0 s, hop 1,0 s (`embeddings.py`, D-56) | WeSpeaker `pyannote/wespeaker-voxceleb-resnet34-LM` sur les mêmes windows (`pretrained_embeddings = True`) |
| Clustering | spectral, nombre de voisins du graphe choisi par fichier (NME-SC, `nme_spectral_clustering`, Laplacien normalisé, D-59), k par eigengap (max 20), KMeans seed 42 | p fixe (`clustering_method = "fixe"`, D-58) ; mode k oracle |
| Post-traitement | vote des windows → frames (`post_treatment.py`, `windows_to_annotation`), avec `fill_unlabeled = True` (trames de parole sans window → label de la trame labellisée la plus proche) | sans `fill_unlabeled` (V2) ; mode vote oracle |
| Pipeline pré-entraîné | — | `pyannote_diarization.py` (`pyannote/speaker-diarization-community-1`) |

Versions de la baseline : V2 = `best_vad` `devtrain` + NME-SC + stats MFCC ; V3 = V2 + `fill_unlabeled`.
Les valeurs des paramètres font foi dans `baseline.py`.

## Résultats de référence (observés)

Protocole : collar 0, UEM = durée du wav, agrégation micro (D-31 à D-34).

### Dev sample 90 (réglage)

| Expérience | Résultat |
|---|---|
| Plancher « un seul locuteur » (dev sample 40) | DER moderne 53,66 % |
| Baseline du 11/11 (energy_vad, p 0.1, max 10) | DER moderne 33,49 % |
| Diagnostic VAD oracle | DER moderne 28,67 % (forgiving 25,99 %) |
| Diagnostic vote oracle | DER moderne 31,63 % (gain uniquement en confusion) |
| Diagnostic k oracle (energy_vad, p 0.1) | DER moderne 38,76 % : **pire** que l'eigengap (33,49 %) |
| Diagnostic k oracle (VAD oracle, p 0.1) | DER 28,67 → 33,06 %, JER 61,13 → 53,46 % : le locuteur dominant est découpé (`results/diag_k_oracle_vs_eigengap/`) |
| `energy_vad` → `vad_stack` (dev-train) | detection error 6,649 % → 3,131 % |
| Clustering p fixe 0.02 (best_vad devtrain) | DER 29,09 %, forgiving 26,21 %, JER 52,91 % |
| NME-SC, Laplacien non normalisé (best_vad devtrain) | DER 26,04 %, forgiving 23,11 %, JER 55,27 % |
| **Baseline V2** (NME-SC normalisé, best_vad devtrain) | DER 27,39 %, forgiving 24,48 %, JER 54,85 % |
| **Baseline V3** (V2 + `fill_unlabeled`) | DER 26,53 %, forgiving 23,57 %, JER 54,47 % |
| **V2 + WeSpeaker** | DER 14,24 %, forgiving 11,02 %, JER 32,08 % (confusion 19,46 → 6,30 %) |
| Baseline du 14/11 (NME-SC normalisé, **best_vad alldev**) | DER 26,43 %, forgiving 23,47 %, JER 53,09 % — **optimiste : VAD entraînée sur ces fichiers** |

pyannote n'a pas été évalué sur le dev sample 90.

### Test sample 90 (mesure)

| Expérience | Résultat |
|---|---|
| Plancher « un seul locuteur » | DER moderne 56,7 % |
| `energy_vad` → `vad_stack` (dev-train) | detection error 11,045 % → 5,761 % |
| Ablation : départ (`energy_vad` + p fixe 0.1) | DER 44,31 %, forgiving 43,16 %, JER 77,19 % |
| Ablation : les deux améliorations (`best_vad` alldev + NME-SC) | DER 30,86 %, forgiving 28,77 %, JER 68,38 % |
| **Baseline V2** | DER 30,77 %, forgiving 28,65 %, JER 67,32 % (FA 2,82 %, missed 6,88 %, confusion 21,07 %) |
| **Baseline V3** | DER 30,07 %, forgiving 27,93 %, JER 67,04 % (FA 3,38 %, missed 5,18 %, confusion 21,51 %) |
| **V2 + WeSpeaker** | DER 17,14 %, forgiving 14,26 %, JER 43,36 % (FA 2,82 %, missed 6,88 %, confusion 7,44 %) |
| **pyannote** `speaker-diarization-community-1` | DER 10,69 %, forgiving 9,51 %, JER 32,55 % (FA 3,62 %, missed 3,58 %, confusion 3,48 %) |

Fichiers sources : `docs/rapport-ablation-et-approches.md` (tableau comparatif, cas d'échec, RTF),
`docs/rapport-ablation.md`, `results/ablation/`, `results/baselineV2 testset/`, `results/baselineV3 testset/`,
`results/baselineV2 pretrained embeddings testset/`, `results/baselineV2 devset/`, `results/baselineV3 devset/`,
`results/pretrained_emb/`, `results/pyannote/`, `results/nme_sc/`, `results/vad_agent/rapport.md`,
`results/diag_k_oracle_vs_eigengap/rapport.md`.

**Lecture actuelle (interprétation)** : avec les stats MFCC, la principale source d'erreur reste la **confusion du
clustering** (21,07 points sur 30,77 pour V2 sur le test), surtout par sous-estimation de k sur les fichiers à nombreux
locuteurs. Les embeddings WeSpeaker, à VAD, windows et clustering identiques, divisent la confusion par 2,8 sur le test :
c'est le levier le plus fort mesuré. pyannote reste devant (10,69 % contre 17,14 %), mais ce passage change plusieurs
composants à la fois.

## Améliorations pour le rendu (D-62)

Ablation fixée le 14/11 (D-62), mesurée sur le test sample 90 (`docs/rapport-ablation.md`) :
- **Amélioration n°1 : `energy_vad` de base (sans hystérésis ni lissage) → `best_vad` (`vad_stack`).** DER −7,42 points
  (sans n°2) et −7,52 (avec n°2) ; 74 fichiers améliorés, 11 dégradés (1 → 2).
- **Amélioration n°2 : `p_neighbors` fixe 0.1 → choisi fichier par fichier (NME-SC, Laplacien normalisé, D-59).**
  DER −5,93 (sans n°1) et −6,03 (avec n°1) ; 39 fichiers améliorés, 18 dégradés (1 → 3).
- Ensemble : 44,31 → 30,86 % ; interaction −0,10 point (niveau du bruit).

## Reste à faire pour le rendu

- [x] Passer `baseline.py` à `best_vad(train="devtrain")` pour les mesures sur le dev (D-57).
- [ ] Passer `baseline_oracle.py` à `devtrain` et aligner ses paramètres de clustering sur `baseline.py` (D-58) : non
  fait ; `baseline_oracle.py` est signalé « pas à jour » dans le README.
- [x] DER, DER forgiving et JER du pipeline complet avec `vad_stack` vs `energy_vad` sans hystérésis : fait dans
  l'ablation (configurations 1 → 2 et 3 → 4).
- [ ] Vérifier la stabilité de l'amélioration n°2 (seeds de KMeans, `max_speakers`) : non fait (une seule seed, 42).
- [x] Ablation sur les 2 améliorations (D-62) : `docs/rapport-ablation.md`.
- [x] Tableau comparatif baseline / variantes / embeddings pré-entraînés / pyannote (DER, JER, RTF, cas d'échec) :
  `docs/rapport-ablation-et-approches.md`.
- [x] Démo : charger un audio, timeline des locuteurs, écoute des segments, export RTTM (`python -m demo.app`).
- [x] README de reproduction et fiche des corpus à jour (16/11).
- [ ] Soutenance.

## Points de vigilance

- **Runs sur le test** : recensés en tête de `docs/rapport-ablation-et-approches.md` (pyannote 07/11, baseline V0
  09/11, VAD seule par `vad-tester`, ablation 14/11, V2 / V3 / V2 + WeSpeaker 16/11). Les runs V0, V2 et V3 sont
  informatifs ; les autres sont comparatifs.
- **Fuite sur le dev** : `baseline_oracle.py` utilise encore `best_vad(train="alldev")` et p 0.1 / max_speakers 10 ; ses
  chiffres sur le dev sont optimistes et non comparables à ceux de `baseline.py`. La ligne « Baseline du 14/11 » du dev
  est optimiste pour la même raison.
- **Modèles pré-entraînés** (D-40) : `pyannote/segmentation-3.0` a vu une partie non précisée de VoxConverse ; WeSpeaker
  est entraîné sur VoxCeleb, qui peut partager des locuteurs avec VoxConverse (non vérifié). Les lignes pyannote et
  V2 + WeSpeaker peuvent être optimistes.
- Une seule seed de KMeans et un seul sample de 90 fichiers : pas d'intervalle de confiance ; les écarts de l'ordre de
  0,1 point (interaction de l'ablation, V2 alldev → devtrain) relèvent du bruit.
- Le gain de NME-SC dépend de la référence choisie (p 0.1 sur le test ; contre p 0.02 sur le dev, DER −1,7 point mais
  JER dégradé).
- Le RTF GPU (pyannote, WeSpeaker) n'est pas représentatif sur ce PC (GPU bloqué en mode idle).
- La démo utilise `best_vad(train="alldev")` (modèles entraînés sur tout le dev).
