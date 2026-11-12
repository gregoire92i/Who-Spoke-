# État du projet — Who Spoke?
 
> Dernière mise à jour : 12/11/2026. Mis à jour par Grégoire, ou par Claude Code uniquement après validation de
> Grégoire (proposition dans le rapport de synchro). Les décisions détaillées sont dans `docs/choix_techniques.md`.
 
## Calendrier
 
- Début : 26/10/2026. Durée : 3 semaines.
- Semaine 1 : corpus et baseline classique (terminée). Semaine 2 : pyannote (fait). **Semaine 3 (en cours)** :
  améliorations mesurées, ablation, démo, documentation, soutenance.
## Pipeline actuel
 
| Étape | Baseline (`baseline.py`) | Variante / alternative |
|---|---|---|
| VAD | `energy_vad` (`audio_features.py`) : seuil relatif au max, hystérésis, smoothing, durées minimales | `vad_stack` / `best_vad` (`vad.py`, subagent `vad-tester`, branche `vad-agent`) |
| Caractéristiques | STFT + MFCC (`audio_features.py`) | — |
| Embeddings | statistiques des MFCC par windows glissantes (`embeddings.py`) | — |
| Clustering | spectral, nombre de locuteurs par eigengap (`clustering.py`) | mode k oracle |
| Post-traitement | vote des windows → frames (`post_treatment.py`, `windows_to_annotation`) | mode vote oracle |
| Pipeline pré-entraîné | — | `pyannote_diarization.py` (`pyannote/speaker-diarization-community-1`) |
 
Les valeurs des paramètres font foi dans `baseline.py`.
 
## Résultats de référence (observés)
 
Protocole : collar 0, UEM = durée du wav, agrégation micro (D-31 à D-34).
 
| Expérience | Fichiers | Résultat |
|---|---|---|
| Plancher « un seul locuteur » | dev sample 40 | DER moderne 53,66 % |
| Plancher « un seul locuteur » | test sample 90 | DER moderne 56,7 % |
| Baseline (sans oracle) | dev sample 90 | DER moderne 33,49 % |
| Diagnostic VAD oracle | dev sample 90 | DER moderne 28,67 % (forgiving 25,99 %) |
| Diagnostic vote oracle | dev sample 90 | DER moderne 31,63 % (gain uniquement en confusion) |
| Diagnostic k oracle | dev sample 90 | DER moderne ≈ 38 % : **pire** que l'eigengap (hausse uniquement en confusion) |
| `energy_vad` → `vad_stack` (dev-train) | dev sample 90 | detection error 6,649 % → 3,131 % |
| `energy_vad` → `vad_stack` (dev-train) | test sample 90 | detection error 11,045 % → 5,761 % |
 
Fichiers sources : `results/` (CSV par fichier, RTTM) et `results/vad_agent/rapport.md`.
 
**Lecture actuelle (interprétation)** : la principale source d'erreur de la baseline est la **confusion du clustering**.
La VAD a une marge de gain d'environ 5 points de DER (VAD oracle).
 
## Améliorations pour le rendu (D-62)
 
- **Amélioration n°1 : VAD apprise `vad_stack`.** Gain mesuré sur la detection error. **Non mesuré** : son effet sur
  le DER du pipeline complet par rapport à `energy_vad`.
- **Amélioration n°2 : non décidée.** Pistes : clustering, taille des windows d'embedding, détection de la musique.
## Reste à faire pour le rendu
 
- [ ] DER, DER forgiving et JER du pipeline complet avec `vad_stack` vs `energy_vad` (mêmes fichiers).
- [ ] Choisir, implémenter et mesurer l'amélioration n°2.
- [ ] Ablation sur les 2 améliorations (mêmes fichiers de validation).
- [ ] Tableau comparatif baseline / variantes / pyannote (DER, JER, RTF, cas d'échec).
- [ ] Démo : charger un audio, timeline des locuteurs, écoute des segments, export RTTM ou JSON.
- [ ] README de reproduction et fiche des corpus à jour.
- [ ] Mettre à jour `docs/choix_techniques.md` dans le dépôt (D-61, D-62 manquent).
- [ ] Soutenance.
## Points de vigilance
 
- Le test a été consulté 4 fois par le subagent `vad-tester` (version fixée avant chaque exécution, selon son rapport).
- Le RTF de pyannote sur ce PC n'est pas représentatif (GPU bloqué en mode idle).