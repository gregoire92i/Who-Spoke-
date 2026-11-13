# Journal — diagnostic k oracle vs eigengap (13/11/2026, subagent clustering-embeddings-analyzer)

Dev sample 90 uniquement (`samples/dev/sample90_seed130.rttm`). Test jamais lu. Aucune modification des sources
(ni temporaire) : tout passe par `analysis/clustering_agent/*.py`, qui importe les modules.

## It. 0 — identifier les configurations
- Lu : `baseline.py`, `baseline_oracle.py` (working tree + `git show HEAD:`), `clustering.py`, `embeddings.py`, `post_treatment.py`, `vad.py` (best_vad).
- Réévaluation des RTTM existants (`eval_existing_rttm.py`) :
  - HEAD (energy_vad ancienne version, p_neighbors=0.1, max_speakers=10) : eigengap 33.49 % → k oracle 38.76 % (confusion 21.55 → 26.82). **Ce sont les chiffres de la question.**
  - working tree (best_vad train="alldev", p=0.1, max=10) : 29.71 % → 33.92 % (confusion 22.63 → 26.84).
  - `baseline_sample90_seed130` working tree : 31.55 % mais p_neighbors=0.05 et max_speakers=20 → **non comparable** aux runs oracle (baseline_oracle.py n'a pas ces paramètres, il utilise les défauts de spectral_clustering).
- Vérifié : dans chaque paire k0/k1, n_windows, n_uncovered et VAD_error identiques par fichier → seul k change.
- Écart au protocole repéré : best_vad(train="alldev") est entraînée sur les 216 fichiers dev, dont le sample 90 évalué (fuite, CLAUDE.md §3).
- Reproduction : VAD oracle eigengap = 28.667 % (identique au CSV committé). energy_vad : 33.95 / 38.27 % au lieu de 33.49 / 38.76 %, car energy_vad a changé au commit 7ade822 (hystérésis, smoothing) après les runs de 4ae8849.

## It. 1 — test sur 3 fichiers (uvnmy, qhesr, usbgm), tag test3
- Le wrapper `spectral_internals` + `labels_for_k` est vérifié par assertion égal à `spectral_clustering` (eigengap et oracle, p=0.1) sur chaque fichier.
- uvnmy (4 réf., dont 2 locuteurs de 2,2 s et 1,8 s) : eigengap k=2 DER 4.8 % ; k=4 DER 56.6 %, le locuteur dominant est coupé en 3.

## It. 2 — sample 90, 3 VAD (oracle, energy, stack_devtrain), p=0.1 (+0.05, 0.02 sous VAD oracle), balayage de k sous VAD oracle
- Fichiers : runs_sample90.csv, speakers_sample90.csv, embeddings_quality_sample90.csv, agregats_micro_sample90.csv, par_fichier_eig_vs_oracle_sample90.csv, sweep_k_sample90.csv, best_k_par_fichier_sample90.csv ; logs aggregate_sample90.log, extra_stats_sample90.log.
- H1 (sur-segmentation du dominant) : confirmée. H3 (locuteurs rares sans bloc dans le graphe kNN) : confirmée. H4 (la VAD retire les windows) : réfutée comme cause. H5 (appariement hongrois) : c'est le canal comptable, pas une cause indépendante. Voir rapport.md.
- Contrôle : un k uniformément plus petit (n_ref−1) est pire que n_ref (36.98 vs 35.69 %, mêmes 79 fichiers) → l'eigengap ne gagne pas parce que son k est « plus petit » mais parce qu'il est choisi fichier par fichier.

## It. 3 — figures (figures/), cas uvnmy, bauzd, hkzpa (contre-exemple où l'oracle aide).

Modifications temporaires de sources : aucune.
