# État du projet — Who Spoke?
 
> Dernière mise à jour : 14/11/2026 (par Claude Code, à la demande de Grégoire). Mis à jour par Grégoire, ou par
> Claude Code uniquement après validation de Grégoire (proposition dans le rapport de synchro). Les décisions détaillées
> sont dans `docs/choix_techniques.md`. Dernier rapport de synchro : `docs/sync/2026-11-13_sync.md`.
 
## Calendrier
 
- Début : 26/10/2026. Durée : 3 semaines.
- Semaine 1 : corpus et baseline classique (terminée). Semaine 2 : pyannote (fait). **Semaine 3 (en cours)** :
  améliorations mesurées, ablation, démo, documentation, soutenance.
## Pipeline actuel
 
| Étape | Baseline (`baseline.py`) | Variante / alternative |
|---|---|---|
| VAD | `best_vad` / `vad_stack` (`vad.py`, VAD apprise, D-57 et D-62) | `energy_vad` sans hystérésis (`audio_features.py`, ancienne baseline et référence de l'amélioration n°1, D-53, D-54) ; VAD oracle (`baseline_oracle.py`) |
| Caractéristiques | STFT (frames 65 ms, hop 32,5 ms) + 13 MFCC (`audio_features.py`) | — |
| Embeddings | statistiques des MFCC (moyenne, écart-type) par windows de 2,0 s, hop 1,0 s (`embeddings.py`, D-56) | — |
| Clustering | spectral, nombre de voisins du graphe choisi par fichier (NME-SC, `nme_spectral_clustering`, Laplacien normalisé, D-59), k par eigengap (max 20) | p fixe `p_neighbors = 0.02` (`clustering_method = "fixe"`, D-58) ; mode k oracle |
| Post-traitement | vote des windows → frames (`post_treatment.py`, `windows_to_annotation`) | mode vote oracle |
| Pipeline pré-entraîné | — | `pyannote_diarization.py` (`pyannote/speaker-diarization-community-1`) |
 
Les valeurs des paramètres font foi dans `baseline.py`.
 
## Résultats de référence (observés)
 
Protocole : collar 0, UEM = durée du wav, agrégation micro (D-31 à D-34). Toutes les lignes sont sur le dev sample 90
sauf mention contraire.
 
| Expérience | Fichiers | Résultat |
|---|---|---|
| Plancher « un seul locuteur » | dev sample 40 | DER moderne 53,66 % |
| Plancher « un seul locuteur » | test sample 90 | DER moderne 56,7 % |
| Baseline du 11/11 (energy_vad, p 0.1, max 10) | dev sample 90 | DER moderne 33,49 % |
| Diagnostic VAD oracle | dev sample 90 | DER moderne 28,67 % (forgiving 25,99 %) |
| Diagnostic vote oracle | dev sample 90 | DER moderne 31,63 % (gain uniquement en confusion) |
| Diagnostic k oracle (energy_vad, p 0.1) | dev sample 90 | DER moderne 38,76 % : **pire** que l'eigengap (33,49 %) |
| Diagnostic k oracle (VAD oracle, p 0.1) | dev sample 90 | DER 28,67 → 33,06 %, JER 61,13 → 53,46 % : le locuteur dominant est découpé (`results/diag_k_oracle_vs_eigengap/`) |
| `energy_vad` → `vad_stack` (dev-train) | dev sample 90 | detection error 6,649 % → 3,131 % |
| `energy_vad` → `vad_stack` (dev-train) | test sample 90 | detection error 11,045 % → 5,761 % |
| Clustering p fixe 0.02 (best_vad devtrain) | dev sample 90 | DER 29,09 %, forgiving 26,21 %, JER 52,91 % |
| NME-SC, Laplacien normalisé (best_vad devtrain) | dev sample 90 | DER 27,39 %, forgiving 24,48 %, JER 54,85 % |
| NME-SC, Laplacien non normalisé (best_vad devtrain) | dev sample 90 | DER 26,04 %, forgiving 23,11 %, JER 55,27 % |
| Baseline actuelle (NME-SC normalisé, **best_vad alldev**, run du 14/11) | dev sample 90 | DER 26,43 %, forgiving 23,47 %, JER 53,09 % (FA 1,34 %, missed 5,73 %, confusion 19,36 %) — **optimiste : VAD entraînée sur ces fichiers** |
 
Fichiers sources : `results/` (CSV par fichier, RTTM), `results/vad_agent/rapport.md`, `results/nme_sc/`,
`results/diag_k_oracle_vs_eigengap/rapport.md`.
 
**Lecture actuelle (interprétation)** : la principale source d'erreur reste la **confusion du clustering**
(environ 18-21 points de DER). Les petits locuteurs ne forment pas de groupe séparé dans le graphe ; DER et JER
réagissent en sens inverse aux changements de k (le DER pénalise le découpage du dominant, le JER l'oubli des petits
locuteurs).
 
## Améliorations pour le rendu (D-62)
 
Ablation fixée le 14/11 (D-62) :
- **Amélioration n°1 : `energy_vad` de base (sans hystérésis ni lissage sur k trames) → `best_vad` (`vad_stack`).**
  Gain mesuré sur la detection error. **Non mesuré** : son effet sur le DER du pipeline complet.
- **Amélioration n°2 : `p_neighbors` fixe → choisi fichier par fichier (NME-SC, Laplacien normalisé, D-59)**, intégré
  dans `baseline.py`. Sur le dev : DER 29,09 → 27,39 %, JER 52,91 → 54,85 %. Stabilité non vérifiée.
## Reste à faire pour le rendu
 
- [ ] Passer `baseline.py` et `baseline_oracle.py` à `best_vad(train="devtrain")` pour les mesures sur le dev (D-57).
- [ ] Aligner les paramètres de clustering de `baseline_oracle.py` sur ceux de `baseline.py` (D-58).
- [ ] DER, DER forgiving et JER du pipeline complet avec `vad_stack` vs `energy_vad` **sans hystérésis** (`delta_db=0, smooth_frames=1`, D-54 révisée le 14/11), mêmes fichiers et même clustering.
- [ ] Vérifier la stabilité de l'amélioration n°2 (seeds de KMeans, `max_speakers`).
- [ ] Ablation sur les 2 améliorations : protocole fixé le 14/11 (D-62) — test sample 90, 4 configurations
  (`energy_vad` −30 dB sans hystérésis ni lissage / `best_vad(train="alldev")` × `p_neighbors` 0.1 fixe / NME-SC),
  `min_speakers` 1, `max_speakers` 20. Pas de passage préalable sur le dev.
- [ ] Tableau comparatif baseline / variantes / pyannote (DER, JER, RTF, cas d'échec).
- [ ] Démo : charger un audio, timeline des locuteurs, écoute des segments, export RTTM ou JSON.
- [ ] README de reproduction (nouveaux scripts : `nme_sc_experiment.py`, `analysis/clustering_agent/`) et fiche des corpus à jour.
- [ ] Soutenance.
## Points de vigilance
 
- Le test a été consulté 4 fois par le subagent `vad-tester` (version fixée avant chaque exécution, selon son rapport).
  Pas d'autre consultation depuis.
- Consultation prévue du test : l'ablation (5e consultation), avec les versions fixées dans D-62 avant exécution.
- **Fuite sur le dev** : `baseline.py` et `baseline_oracle.py` utilisent `best_vad(train="alldev")`, entraînée sur les
  90 fichiers du sample de réglage : leurs chiffres sur le dev sont optimistes (D-57).
- `baseline_oracle.py` utilise encore p 0.1 / max_speakers 10 : ses runs ne sont pas comparables à ceux de `baseline.py`.
- Le gain de NME-SC repose sur quelques fichiers (autant de fichiers dégradés que d'améliorés) et dégrade le JER.
- Le RTF de pyannote sur ce PC n'est pas représentatif (GPU bloqué en mode idle).
