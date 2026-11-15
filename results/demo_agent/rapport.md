# Rapport du subagent `demo-builder` : démo Who Spoke?

> Rédigé par Claude Code le 14/11/2026, à la demande de Grégoire. Branche `demo`, dépôt principal, aucun commit.
> Fichiers audio utilisés : **dev VoxConverse uniquement** (et des fichiers dérivés du dev : extraits, conversions de
> format). Le test VoxConverse n'a jamais été utilisé. Journal détaillé : `results/demo_agent/journal.md`.

## 1. Documents lus

`CLAUDE.md` ; `docs/enonce_projet.md` (vrai énoncé, après correction par Grégoire) ; `docs/etat_du_projet.md` ;
`docs/choix_techniques.md` (D-05, D-15, D-40, D-50, D-57, D-61, D-62, décisions ouvertes) ; `docs/rapport-ablation.md` ;
`results/vad_agent/rapport.md` ; `README.md` ; code de `baseline.py`, `vad.py`, `pyannote_diarization.py`,
`post_treatment.py`, signatures d'`audio_features.py`, `embeddings.py`, `clustering.py`, en-tête d'`ablation.py` ;
`.gitignore`, `requirements.txt`. Non lus : `docs/notes-importantes-et-difficultes.md`, `docs/sync/`.

## 2. Choix faits par Grégoire (phase 1, recopiés)

1. Baseline : **B3**, `baseline.diarize_file(..., vad_train="devtrain")` ajouté par Grégoire ; la démo appelle
   `vad_train="alldev"`, `frame_size=baseline.frame_size`, `hop_size=baseline.hop_size`. Aucun remplacement de
   fonction à l'exécution. Test d'égalité avec `vad_train="devtrain"` sur 1-2 fichiers dev courts.
2. Modèles `best_vad` manquants : bandeau d'information + réentraînement si les prérequis sont là ; sinon baseline
   désactivée avec les commandes, pyannote reste disponible.
3. Pipelines : baseline + pyannote (sélecteur), sans variantes.
4. Interface **Gradio 6.32.0**, timeline **plotly** ; installations autorisées ; pytest autorisé.
5. pyannote : `HF_HUB_OFFLINE=1` avant l'import, sans jeton ; GPU si disponible, sinon CPU (affiché).
6. `.gitignore` : `demo/tmp/`, `.gradio/`, `*.m4a *.ogg *.opus *.aac *.wma *.webm`.
7. Export **RTTM uniquement**. Optionnelles a (référence + DER/JER), b (comparaison), c (micro), d (renommage) = oui
   après les obligatoires ; e (tous les segments d'un locuteur), f (marge d'écoute) = non.

## 3. Architecture

```
demo/
  __init__.py          fixe HF_HUB_OFFLINE=1, GRADIO_ANALYTICS_ENABLED=False, GRADIO_TEMP_DIR=demo/tmp/gradio,
                       MPLBACKEND=Agg AVANT tout autre import ; ROOT, TMP_DIR
  core.py              indépendant de l'interface : load_audio, Engines, diarize, speaker_stats, segments_table,
                       segment_audio, rename, to_rttm, load_reference, evaluate
  baseline_adapter.py  appelle baseline.diarize_file(path, uri, frame_size=…, hop_size=…, vad_train="alldev") ;
                       check_vad_models / prepare (modèles best_vad) ; params() lus dans baseline.py et vad.py
  pyannote_adapter.py  réutilise pyannote_diarization.load_pipeline et diarize_file (sans chauffe)
  plots.py             timeline et barres plotly (interface) ; timeline matplotlib pour figures/demo/
  app.py               interface Gradio (python -m demo.app)
  tmp/                 ignoré par git : wav convertis, uploads Gradio, exports RTTM
```

- **Aucun paramètre de pipeline n'est défini dans `demo/`** : la baseline lit tout dans `baseline.py`
  (frames 65 / 32,5 ms, 13 MFCC, windows 2,0 / 1,0 / 0,5 s, NME-SC normalisé, min/max speakers 1 / 20, seed 42…) ;
  la VAD est `best_vad` avec `PROD_PARAMS` de `vad.py` ; pyannote suit `pyannote_diarization.py` (`oracle=False`).
- **Chargement audio** : wav et flac sont passés tels quels aux pipelines (la baseline les traite donc exactement
  comme `baseline.py`) ; tout autre format (mp3, ogg, m4a, aac…) est d'abord converti en wav 16 kHz mono PCM 16 bits
  par ffmpeg dans `demo/tmp/converted/` (D-50). Le signal 16 kHz mono (librosa) sert à l'écoute ; la durée est
  `len(y) / 16000`.
- **Modèles chargés une fois** (`core.Engines`) : `best_vad` alldev (modèles lus par `vad.get_model` et `vad.get_stack_model`,
  gardés en mémoire par `vad._MODELS`, sans calcul sur un signal) et la pipeline pyannote (GPU si `torch.cuda.is_available()`, sinon CPU ; `WHO_SPOKE_DEVICE=cpu`
  force le CPU). Si une pipeline ne se charge pas, l'autre reste disponible.
- **Confidentialité (D-05)** : `launch(server_name="127.0.0.1", share=False, ssr_mode=False, mcp_server=False)`,
  analytics désactivées, aucun jeton lu ni affiché par la démo (pyannote hors ligne). Port vérifié :
  `LISTEN 127.0.0.1:7861` uniquement. Une seule analyse à la fois (`queue(default_concurrency_limit=1)`).
- URI : nom du fichier sans extension, accents retirés, caractères hors `[A-Za-z0-9_-]` → `_`
  (« réunion inconnue.m4a » → `reunion_inconnue`).

## 4. Fonctionnalités

| Fonctionnalité | État | Vérification |
|---|---|---|
| 1. Charger un audio (wav, mp3, m4a, flac, ogg, opus, aac, webm ; formats affichés) | livrée | API (gradio_client) + tests + cas limites |
| 2. Lancer l'analyse, progression et temps écoulé | livrée (`gr.Progress` ; Gradio affiche le chronomètre) | API ; affichage de la progression non vu dans un navigateur |
| 3. Timeline : une ligne par locuteur, couleur fixe (Okabe-Ito, hachures au-delà de 8), ligne `overlap`, axe en s, survol en m:ss, zoom souris + rangeslider si > 120 s | livrée | figure plotly produite (JSON vérifié) ; rendu non vu dans un navigateur ; équivalent matplotlib dans `figures/demo/` |
| 4. Temps de parole : tableau (s, m:ss, % de la parole, % de l'audio, nb de segments) + barres | livrée | API + test `test_speaker_stats_consistent` |
| 5. Tableau des segments, clic sur une ligne → lecture | livrée (`Dataframe.select`, `SelectData.row_value`) | `segment_audio` testé ; **le clic n'a pas été testé de bout en bout** (pas de navigateur) |
| 6. Export RTTM (téléchargement) | livrée (`Annotation.write_rttm`) | API (fichier reçu) + tests aller-retour |
| Infos : pipeline, paramètres, durée, temps, RTF, appareil | livrée | API |
| a. RTTM de référence → timelines réf. + DER, DER forgiving, JER, FA / missed / confusion | livrée | API sur afjiv (dev) ; test réf. vs elle-même = 0 |
| b. Comparaison baseline / pyannote (2 timelines, sélecteur « Résultats affichés ») | livrée | API sur afjiv |
| c. Micro dans le navigateur (`gr.Audio(sources=["microphone"])`) | livrée, **non testée** (pas de navigateur ni de micro dans cette session) | — |
| d. Renommage des locuteurs (doublons refusés, espaces → `_`, appliqué au RTTM) | livrée | API (RTTM avec `Alice_Martin`) + test |
| e, f | non livrées (décision) | — |

Aucun chiffre de performance n'est affiché sans référence : DER / JER n'apparaissent que si un RTTM de référence est
chargé, avec la mention « ce fichier uniquement ».

## 5. Tests (sortie exacte)

Commande : `python -m pytest tests/test_demo.py -v` (depuis la racine, `.venv` activé).

```
collecting ... collected 16 items

tests/test_demo.py::test_rttm_roundtrip_synthetic PASSED                 [  6%]
tests/test_demo.py::test_rttm_roundtrip_pipeline_output PASSED           [ 12%]
tests/test_demo.py::test_rttm_empty_annotation PASSED                    [ 18%]
tests/test_demo.py::test_make_uri PASSED                                 [ 25%]
tests/test_demo.py::test_speaker_stats_consistent PASSED                 [ 31%]
tests/test_demo.py::test_speaker_stats_empty PASSED                      [ 37%]
tests/test_demo.py::test_segment_audio_bounds PASSED                     [ 43%]
tests/test_demo.py::test_load_audio_formats PASSED                       [ 50%]
tests/test_demo.py::test_load_audio_errors PASSED                        [ 56%]
tests/test_demo.py::test_adapter_equals_baseline_devtrain[hqyok] PASSED  [ 62%]
tests/test_demo.py::test_adapter_equals_baseline_devtrain[tucrg] PASSED  [ 68%]
tests/test_demo.py::test_demo_uses_alldev PASSED                         [ 75%]
tests/test_demo.py::test_vad_models_present PASSED                       [ 81%]
tests/test_demo.py::test_evaluate_reference_against_itself PASSED        [ 87%]
tests/test_demo.py::test_rename PASSED                                   [ 93%]
tests/test_demo.py::test_vad_models_missing_states PASSED                [100%]

============================== 16 passed in 5.94s ==============================
```

Points à connaître :
- **Aller-retour RTTM sur une vraie sortie de la baseline (afjiv)** : le DER n'est **pas exactement 0**
  (8,43e-05 observé au premier essai) : `write_rttm` écrit 3 décimales, les frontières de la baseline sont des
  multiples de 32,5 ms, et début et durée sont arrondis séparément. Le test vérifie : mêmes labels, écart ≤ 0,5 ms
  sur les débuts et ≤ 1 ms sur les fins, DER < 1e-3, puis un second aller-retour exact (DER = 0, collar 0,
  moderne et forgiving). Sur une annotation synthétique (frontières à la ms), DER = 0 directement.
- **Égalité avec `baseline.py`** : sur hqyok et tucrg (dev), les pistes de l'adaptateur avec `vad_train="devtrain"`
  sont identiques à `baseline.diarize_file(..., vad_train="devtrain")` ; sur hqyok, `core.diarize` (démo complète)
  est identique à `baseline.diarize_file(..., vad_train="alldev")`.
- Modèles absents : simulé (dossier de modèles vide) sans toucher aux vrais `.pkl` : état `retrain` sur ce PC
  (prérequis présents), `missing` si la racine est vide (message avec la commande), baseline indisponible.
- Les tests qui lisent l'audio dev sont sautés si `data/voxconverse/dev/` est absent.

## 6. Cas limites (observé)

Matériel : RTX 3050 Laptop 4 Go, 12 cœurs, 7 Go de RAM, WSL2. Script hors dépôt (scratchpad), fichiers générés dans
`demo/tmp/` puis supprimés. Fichiers dérivés : extraits et conversions de afjiv (dev, 151,2 s, 5 locuteurs de réf.).

| Cas | Comportement observé | Statut |
|---|---|---|
| Audio 1,5 s (extrait afjiv) | baseline k=1, 1 segment ; pyannote k=1, 1 segment. Fins de segment au-delà de la durée : 1,5275 s (baseline) et 1,5497 s (pyannote) pour 1,50 s | OK ; dépassement < 0,05 s (prolongement `d_end` de `best_vad` pour la baseline), visible dans le RTTM |
| Audio 0,3 s | baseline k=0, RTTM vide ; pyannote k=1, 0,4 s de parole (fin 0,402 s) | OK, message « aucune parole » pour la baseline |
| Silence 30 s | baseline et pyannote : k=0, RTTM vide (0 ligne) | OK |
| Bruit blanc faible 30 s | baseline et pyannote : k=0, RTTM vide | OK |
| Un seul locuteur (abjxc, 68,4 s, 1 locuteur de réf.) | baseline **k=2** ; pyannote k=1 | fonctionne ; erreur de la baseline (montrée telle quelle) |
| Musique (pqmho, 139,0 s) | baseline 13,1 s de parole, k=1 ; pyannote 15,8 s, k=1 | fonctionne |
| Stéréo 44,1 kHz flac, nom avec espaces | lu directement ; baseline k=4 ; pyannote k=5, overlap 0,3 s ; URI `afjiv_stereo_44k` | OK |
| `.m4a` | converti par ffmpeg ; baseline k=4 ; pyannote k=5 | OK |
| `.mp3` 48 kHz | **avant correction** : pyannote `ValueError: requested chunk [ 00:00:00.000 -->  00:00:10.000] from afjiv file resulted in 478895 samples instead of the expected 480000 samples.` ; **après** (conversion en wav) : baseline k=4, pyannote k=5 (43 segments contre 24 pour le flac) | corrigé |
| Fichier corrompu (en-tête RIFF + zéros) | `AudioLoadError` : « Format non supporté ou fichier corrompu (corrupt.wav) : Error opening input files: Invalid data found when processing input » ; message Gradio, pas de crash | OK |
| Texte renommé en `.m4a` / `.mp3`, fichier vide, fichier absent | `AudioLoadError` (test `test_load_audio_errors`) | OK |
| Aucun fichier fourni | message « Choisir un fichier audio (ou enregistrer au micro)… » | OK |
| Fichier long (ldnro, 1097,3 s, 15 locuteurs de réf.) | baseline 11,05 s (RTF 0,010), k=9 ; pyannote GPU 291,43 s (RTF 0,266), k=13, overlap 19,8 s ; **pic de RAM du processus 2 854 584 ko (≈ 2,7 Go)**, les deux pipelines chargées | OK sur 7 Go |

## 7. Temps observés

| Mesure | Fichier | Matériel | Résultat |
|---|---|---|---|
| Démarrage à froid (1er lancement de la session, imports compris) | — | GPU | `Engines` 23,2 s (dont pyannote 19,9 s) |
| Démarrage (répétition, terminal propre `env -i`, lancement → page servie) | — | GPU | 11,25 s (2 lancements) ; modèles prêts en 7,4 s et 7,7 s |
| Baseline | afjiv 151,2 s | CPU | 0,8 s, RTF 0,005 |
| Baseline | « réunion inconnue.m4a » (cqaec dev, 224,6 s, converti en stéréo 44,1 kHz m4a) | CPU | 1,0 s, RTF 0,004 ; analyse complète vue du client 1,7 s |
| Baseline | ldnro 1097,3 s | CPU | 11,05 s, RTF 0,010 |
| pyannote | afjiv 151,2 s | GPU | 39,7 s, RTF 0,262 (et 0,257 lors de la génération des figures) |
| pyannote | « réunion inconnue.m4a » 224,6 s | GPU | 58,4 s, RTF 0,260 ; vue du client 59,3 s |
| pyannote | ldnro 1097,3 s | GPU | 291,43 s, RTF 0,266 |
| pyannote | afjiv 151,2 s | **CPU** (`WHO_SPOKE_DEVICE=cpu`) | 143,1 s, **RTF 0,946** |

- Le RTF GPU de pyannote sur ce PC n'est pas représentatif (GPU bloqué en mode idle, `CLAUDE.md`).
- Le RTF inclut la lecture du fichier par la pipeline (comme `baseline.py`), pas la conversion ffmpeg.
- Mesures de cette section faites avec l'ancienne version, qui comportait une chauffe au démarrage (supprimée depuis
  par décision de Grégoire) ; voir la section « Tests du RTF » pour les mesures sans chauffe.
- Résultat observé (non interprété plus loin) : sur la « réunion inconnue » (5 locuteurs de référence dans le dev),
  la baseline trouve k=1 (aussi sur le wav d'origine : k=1) et pyannote k=3.

Figures (matplotlib, dev, avec la référence) : `figures/demo/timeline_afjiv_baseline.png` (k=4, DER 18,0 %,
JER 35,5 %), `figures/demo/timeline_afjiv_pyannote.png` (k=5, DER 9,7 %, JER 10,3 %),
`figures/demo/timeline_ldnro_baseline.png` (k=9, DER 27,4 %, JER 62,2 %). Ce sont des fichiers isolés : pas des
résultats de référence (et pyannote a pu voir le dev, D-40). Aucune capture d'écran de l'interface : pas de navigateur
disponible dans cette session.

## 8. Risques pour la soutenance et plans B

| Risque | Plan B |
|---|---|
| Interface jamais vue dans un navigateur par l'agent (clic sur un segment, rendu plotly, micro) | **Répétition par Grégoire dans son navigateur avant le jour J** (15 min) ; signaler tout défaut |
| Pas de réseau | rien à faire : pyannote hors ligne vérifié (cache `~/.cache/huggingface/hub` requis sur la machine) |
| Machine différente sans le cache pyannote | baseline seule (bandeau « pyannote indisponible ») ; ou copier `~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1` et ses dépendances avant |
| GPU indisponible | pyannote passe en CPU automatiquement : RTF 0,946 observé (≈ 1 min de calcul par minute d'audio) → n'utiliser pyannote que sur des extraits courts, ou la baseline (RTF ≈ 0,01) |
| Enregistrement long (15-20 min) | baseline ≈ 11 s ; pyannote GPU ≈ 5 min (ldnro) : lancer pyannote en dernier ou sur un extrait ; RAM OK (≈ 2,7 Go) |
| Commande lancée hors de la racine | `ModuleNotFoundError: No module named 'demo'` (observé) → `cd` à la racine |
| Port 7860 occupé | `WHO_SPOKE_PORT=7870 python -m demo.app` |
| Navigateur Windows et WSL2 | ouvrir `http://127.0.0.1:7860` ; la redirection localhost de WSL2 est à vérifier avant |
| Format exotique (téléphone) | conversion ffmpeg automatique ; fichier illisible → message, pas de crash |
| Modèles `best_vad` absents | improbable (`.pkl` versionnés) ; sinon réentraînement automatique (≈ 3-5 min, prérequis : cache dev + annotations dev) ou baseline désactivée avec la commande |
| Version de scikit-learn différente | les `.pkl` peuvent ne plus se relire : garder `scikit-learn==1.9.1` (`requirements.txt`) |
| Erreurs de la baseline visibles en direct (k sous-estimé, ex. k=1 sur la réunion de 5 locuteurs) | c'est le résultat honnête ; la comparaison avec pyannote (option b) le montre |

## 9. Textes proposés (à intégrer par Grégoire)

### README, section « Lancer la démo »

```markdown
## Lancer la démo

Application locale (Gradio) : charger un audio, lancer la diarisation (baseline ou pyannote), afficher la timeline
des locuteurs et leur temps de parole, écouter les segments, exporter le RTTM.

    source .venv/bin/activate
    python -m demo.app          # depuis la racine du dépôt

Puis ouvrir http://127.0.0.1:7860 dans le navigateur. L'application n'écoute que sur 127.0.0.1 (aucun lien public,
statistiques Gradio désactivées) ; les fichiers déposés, convertis et exportés restent dans `demo/tmp/` (ignoré par git).

- Formats : wav, flac (lus directement), mp3, m4a, ogg, opus, aac, webm (convertis en wav 16 kHz mono par ffmpeg).
- Baseline : `baseline.diarize_file` avec la VAD `best_vad` de production (`train="alldev"`, modèles dans
  `results/vad_agent/models/`) ; paramètres lus dans `baseline.py`.
- pyannote : `pyannote/speaker-diarization-community-1`, chargé hors ligne depuis le cache Hugging Face
  (`HF_HUB_OFFLINE=1`) ; le modèle doit avoir été téléchargé une fois (voir `pyannote_diarization.py`).
  GPU si disponible, sinon CPU (`WHO_SPOKE_DEVICE=cpu` pour forcer le CPU).
- Options : RTTM de référence (DER, DER forgiving, JER sur ce fichier, collar 0), comparaison des deux pipelines,
  enregistrement au micro, renommage des locuteurs.
- Port : `WHO_SPOKE_PORT=7870 python -m demo.app`.

Tests : `python -m pytest tests/test_demo.py -v`
```

Arborescence à ajouter : `demo/` (application de démonstration), `tests/` (tests pytest de la démo).

### `docs/etat_du_projet.md`

- « Reste à faire » : remplacer « Démo : charger un audio, timeline des locuteurs, écoute des segments, export RTTM
  ou JSON » par « [x] Démo (`demo/`, Gradio, export RTTM uniquement) : 16 tests passés ; reste une répétition dans le
  navigateur (clic sur les segments, micro) ».
- « Points de vigilance » : ajouter « Démo : pyannote sur CPU, RTF 0,946 observé (afjiv, dev) ; sur GPU, RTF ≈ 0,26 ;
  baseline RTF ≈ 0,01. »

Décision à enregistrer éventuellement (proposition, pas de numéro attribué) : « Démo : Gradio 6.32 + plotly, local
uniquement, baseline avec `best_vad(train="alldev")`, pyannote hors ligne, export RTTM seul ; formats autres que
wav/flac convertis par ffmpeg (pyannote échoue sur la lecture directe d'un mp3) ».

## 10. Dépendances ajoutées

- pytest 9.2.0 (+ iniconfig 2.3.1, pluggy 1.7.0) : installés en phase 1 (déjà présents dans `requirements.txt`
  du commit 385bc84).
- gradio 6.32.0 et plotly 7.1.0, avec 24 dépendances ; 26 lignes ajoutées à `requirements.txt` (format `pip freeze`,
  aucune ligne existante modifiée) :
  `annotated-doc==0.0.6, annotated-types==0.8.0, audioop-lts==0.2.2, brotli==1.2.1, fastapi==0.162.7, gradio==6.32.0,
  gradio_client==2.7.2, groovy==0.1.2, hf-gradio==0.4.1, httpcore==1.0.9, httpx==0.28.1, orjson==3.13.0,
  plotly==7.1.0, pydantic==2.14.3, pydantic_core==2.50.4, pydub==0.25.1, python-multipart==0.0.33, pytz==2026.6,
  safehttpx==0.1.7, semantic-version==2.10.0, shellingham==1.5.4, starlette==1.10.0, tomlkit==0.14.0, typer==0.29.1,
  typing-inspection==0.4.5, uvicorn==0.54.0`.
- Non installé : kaleido (export d'images plotly) ; les figures du rapport passent par matplotlib.

## 11. Fichiers créés / modifiés (provenance : C)

Créés :
- `demo/__init__.py`, `demo/core.py`, `demo/baseline_adapter.py`, `demo/pyannote_adapter.py`, `demo/plots.py`,
  `demo/app.py` ;
- `tests/test_demo.py` ;
- `figures/demo/timeline_afjiv_baseline.png`, `figures/demo/timeline_afjiv_pyannote.png`,
  `figures/demo/timeline_ldnro_baseline.png` ;
- `results/demo_agent/journal.md`, `results/demo_agent/rapport.md`.

Modifiés : `.gitignore` (11 lignes validées), `requirements.txt` (26 lignes).

Non modifiés : `baseline.py` (modification B3 faite par Grégoire), `vad.py`, `audio_features.py`, `embeddings.py`,
`clustering.py`, `post_treatment.py`, `pyannote_diarization.py`, `README.md`, `docs/`.

## 12. Commandes

```bash
source .venv/bin/activate
python -m demo.app                          # démo sur http://127.0.0.1:7860 (depuis la racine)
WHO_SPOKE_DEVICE=cpu python -m demo.app     # forcer pyannote sur CPU
WHO_SPOKE_PORT=7870 python -m demo.app      # autre port
python -m pytest tests/test_demo.py -v      # tests (16)
```

## 13. Embellissement de l'interface (ajout du 14/11)

- Fichiers : `demo/app.py` et `demo/plots.py` modifiés, `demo/style.css` créé. Comportement, pipelines, `core.py` et
  adaptateurs inchangés. Aucune dépendance ni ressource téléchargée (polices système).
- Thème `gr.themes.Soft`, accent #0072B2, coins arrondis ; en-tête en dégradé bleu avec titre, sous-titre et pastilles
  d'état des pipelines ; sections numérotées (1 Audio, 2 Analyse, 3 Timeline, 4 Temps de parole, 5 Segments, écoute
  et export) ; cartes de chiffres clés (durée, locuteurs, parole / overlap, traitement, RTF, appareil) ; légende des
  locuteurs ; tableau des métriques de référence ; colonne « locuteur » des tableaux colorée avec la couleur du
  locuteur (texte noir ou blanc selon la luminance) ; figures plotly à fond transparent, texte gris moyen #8b93a1,
  grille discrète, coins arrondis des segments, référence en gris.
- Vérifié hors ligne (espace de noms réseau sans réseau) côté serveur ; rendu navigateur (clair / sombre) **non vu**.
- Tests : `16 passed in 6.39s`.

## 14. Tests du RTF (15/11, sans chauffe)

**Question** : pourquoi le RTF de la baseline affiché par la démo (≈ 0,013 sur un enregistrement de 13 s en m4a)
est-il plus élevé que celui de `baseline.py` (médiane 0,0049 dans `results/baseline_sample90_seed130_par_fichier.csv`,
90 fichiers dev, durée médiane 241,8 s) ?

**Conditions** : i5-11400H (6 cœurs / 12 threads), 7 Go de RAM visibles par WSL2, RTX 3050 Laptop (GPU bloqué en mode
idle : RTF pyannote non représentatif) ; scikit-learn 1.9.1, OpenBLAS et OpenMP à 12 threads par défaut. Machine
**pas forcément au repos** (session WSL interactive). Code de la démo **sans chauffe** (état du 14/11 au soir) ; un
processus neuf par série (= un lancement de l'interface, les deux pipelines chargées comme `demo.app.main`), puis
`core.load_audio` + `core.diarize` répétés. Fichiers **dev** uniquement : afjiv (extrait 80-93 s et entier 151,2 s),
ipqqq (extraits 0-5 … 0-300 s), qzwxa (224 s, proche de la durée médiane du sample 90), ldnro (1097,3 s). Audio dérivés
dans le scratchpad, supprimés. « Appels 2+ » = médiane des appels après le 1er du processus.
Les CSV du premier agent (avec l'ancienne version) sont gardés dans `rtf_tests/ancien_premier_agent/` : leurs valeurs
« appels 2+ » sont du même ordre, mais ils étaient incomplets (pyannote : durées limitées à 5 s, 2 appels sur
afjiv entier ; ni décomposition, ni test de chronométrage, ni test par l'API) et sont remplacés.

### Résultats observés

**A. Même fichier, même processus : `baseline.py` vs démo** (`compare_timing_summary.csv`, 8 appels entrelacés)

| Fichier | `diarize_file` devtrain (= `baseline.main`) | `diarize_file` alldev | `core.diarize` (démo) |
|---|---|---|---|
| afjiv 13 s wav | 0,1405 s, RTF 0,0108 (min 0,0794, max 0,1477) | 0,1474 s, RTF 0,0113 | 0,1511 s, RTF 0,0116 |
| qzwxa 224 s | 0,8949 s, RTF 0,0040 | 0,8709 s, RTF 0,0039 | 0,9321 s, RTF 0,0042 |

`baseline_adapter.params()` : 0,0000 s. Même chronométrage dans les deux cas : `baseline.main` chronomètre
`diarize_file` (lecture du fichier comprise), `core.diarize` aussi.

**B. Format** (`formats_summary.csv`, 1er appel + 3 appels, processus neuf par format)

| afjiv 13 s, baseline | load_audio (conversion comprise) | core.diarize | RTF affiché | RTF total | 1er appel : RTF affiché |
|---|---|---|---|---|---|
| wav | 0,0032 s | 0,1567 s | 0,0121 | 0,0123 | 0,0305 |
| flac | 0,0080 s | 0,1856 s | 0,0143 | 0,0149 | 0,0377 |
| mp3 | 0,1143 s | 0,1756 s | 0,0135 | 0,0223 | 0,0271 |
| m4a | 0,1042 s | 0,1623 s | 0,0124 | 0,0204 | 0,0272 |

afjiv entier (151,2 s), baseline, RTF affiché : wav 0,0051, flac 0,0057, mp3 0,0047, m4a 0,0046 ; load_audio m4a
0,191 s. pyannote 13 s : wav 0,0925, flac 0,0902, mp3 0,0908, **m4a 0,1178** ; wav allongé à 13,056 s (durée du m4a
décodé) : **0,1173**. pyannote entier : 0,2569 à 0,2596 quel que soit le format.

**C. Répétitions** (`repeat.csv`, 10 analyses d'afjiv 13 s dans un même processus)

| Série | RTF affiché, appels 1 à 10 | Appels 2-10 : médiane [min ; max] | Tendance appels 2-10 |
|---|---|---|---|
| baseline wav | 0,0237 0,0104 0,0125 0,0106 0,0129 0,0093 0,0121 0,0118 0,0119 0,0128 | 0,0119 [0,0093 ; 0,0129] | pente +0,00016/appel, Spearman ρ +0,28 (p 0,46) |
| baseline m4a | 0,0221 0,0112 0,0101 0,0108 0,0107 0,0108 0,0058 0,0052 0,0126 0,0106 | 0,0107 [0,0052 ; 0,0126] | pente −0,00019/appel, ρ −0,22 (p 0,58) |
| pyannote wav | 0,1779 puis 0,0899 à 0,0907 | 0,0903 | ρ −0,62 (p 0,08) |
| pyannote m4a | 0,2021 puis 0,1159 à 0,1207 | 0,1170 | ρ −0,43 (p 0,24) |

**D. Par l'interface** (`e2e.csv`, `gradio_client`, serveur sur 127.0.0.1:7870, arrêté par son PID) : baseline afjiv
13 s m4a, 10 analyses : RTF affiché 0,031 (1re), puis 0,014, 0,012 ×7, 0,011 ; temps client 1,683 s (1re) puis
0,399 à 0,456 s ; en-tête « analyse terminée en » 1,5 s puis 0,2-0,3 s. Même fichier en wav : RTF 0,011 à 0,016,
temps client 0,297 à 0,400 s. pyannote m4a : 0,235 puis 0,118 ×4 (client 3,27 s puis 1,75 s).

**E. RTF selon la durée** (`durations_summary.csv`, `durations_fit.csv`, figure `figures/demo/rtf_selon_duree.png`)

| Durée (ipqqq) | 5 s | 10 s | 13 s | 20 s | 30 s | 60 s | 120 s | 300 s | ldnro 1097 s |
|---|---|---|---|---|---|---|---|---|---|
| baseline, appels 2+ | 0,0065 | 0,0161 | 0,0190 | 0,0089 | 0,0079 | 0,0054 | 0,0046 | 0,0043 | 0,0082 |
| baseline, 1er appel | 0,0436 | 0,0261 | 0,0310 | 0,0210 | 0,0139 | 0,0096 | 0,0066 | 0,0062 | 0,0140 |
| pyannote, appels 2+ | 0,0735 | 0,0371 | 0,0901 | 0,1513 | 0,1914 | 0,2314 | 0,2509 | 0,2631 | 0,2694 (2 appels) |
| pyannote, 1er appel | 0,3315 | 0,1479 | 0,1791 | 0,2070 | 0,2375 | 0,2536 | 0,2617 | 0,2671 | 0,2700 |

Ajustement linéaire (appels 2+, extraits 5-300 s) : baseline **0,099 s + 0,00386 s par seconde d'audio** ;
pyannote −2,12 s + 0,2707 s/s (pas linéaire sous 30 s : palier 0,37 s jusqu'à 10 s, puis +≈0,36 s par seconde
d'audio en plus jusqu'à 20 s). Baseline sur ldnro : 9,02 s (RTF 0,0082), au-dessus de la droite (clustering 4,56 s).

**F. Décomposition de la baseline** (`stages_summary.csv`, médiane des appels 2-4 ; mêmes fonctions et paramètres que
`diarize_file`, sortie identique à `diarize_file` vérifiée sur les 3 fichiers)

| Étape | afjiv 13 s | qzwxa 224 s | ldnro 1097 s |
|---|---|---|---|
| chargement (librosa) | 0,0022 | 0,0385 | 0,3238 |
| STFT | 0,0056 | 0,0611 | 0,3060 |
| VAD : caractéristiques (`vad.compute_features`) | 0,0428 | 0,5607 | 3,4725 |
| VAD : modèles (`vad.vad_stack`) | **0,0769** | 0,0800 | 0,4777 |
| MFCC | 0,0139 | 0,0268 | 0,0797 |
| embeddings | 0,0006 | 0,0122 | 0,0630 |
| clustering (NME-SC) | 0,0006 | 0,2688 | 4,5576 |
| post-traitement | 0,0006 | 0,0042 | 0,0180 |
| **total** | **0,1475** | **1,0624** | **9,4045** |

1er appel du processus sur 13 s : 0,780 s (chargement 0,350, STFT 0,084, caractéristiques VAD 0,291).

**G. Threads** (`threads_summary.csv`, `vadstack_threads.txt`) : `vad_stack` sur 13 s, 30 appels : médiane 0,0693 s,
**bimodal** (0,023 à 0,094 s) avec 12 threads ; **0,0164 s** (0,016-0,019) avec `OMP_NUM_THREADS=1`. Sur 224 s :
0,0899 s (12 threads) contre 0,1781 s (1 thread). Démo complète (appels 2-10, variables d'environnement seulement,
aucun code modifié) :

| Fichier | 12 threads (défaut) | `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` | 1 thread |
|---|---|---|---|
| afjiv 13 s m4a | 0,1452 s, **RTF 0,0111** | 0,0517 s, **RTF 0,0040** | 0,0529 s, RTF 0,0041 |
| qzwxa 224 s | 0,9546 s, RTF 0,0043 | 0,7852 s, RTF 0,0035 | 0,8637 s, RTF 0,0039 |
| ldnro 1097 s (3 appels) | 12,31 s, RTF 0,0112 (9,24 à 16,82) | 6,32 s, RTF 0,0058 | — |

RTTM de la démo identiques octet par octet (`cmp`) avec 4 threads et par défaut sur afjiv 13 s m4a, qzwxa et ldnro.

### Interprétation

Écart à expliquer : ≈ 0,012 (13 s m4a, appels 2+) contre 0,0049 (`baseline.py`, fichiers de plusieurs minutes),
soit ≈ 0,007 à 0,008 de RTF. Causes, par part de l'écart :
1. **Coût fixe sur un fichier court, dû presque entièrement au parallélisme (12 threads OpenMP / OpenBLAS)** :
   ≈ 0,007 sur ≈ 0,0075 (≈ 90-95 %). Sur 13 s, `vad_stack` coûte 0,077 s (autant que sur 224 s) et les
   caractéristiques 0,043 s, contre 0,016 s et 0,027 s en 1 thread. Avec 4 threads, le 13 s m4a tombe à RTF 0,0040,
   au niveau des fichiers longs. Ce coût existe aussi dans `baseline.py` (même code), mais il est dilué sur des
   fichiers de plusieurs minutes (les plus courts du sample 90, 26 s et 53 s, y ont déjà 0,0055 et 0,0073).
2. **Bruit de mesure** : sur 13 s, d'un appel à l'autre le RTF varie de 0,0052 à 0,0129 (temps bimodal de
   `vad_stack`). Il ne crée pas l'écart moyen mais explique les différences entre deux lancements.
3. **Différence de chronométrage `core.diarize` / `baseline.py`** : ≤ 0,0008 (0,0116 contre 0,0108 sur 13 s ;
   0,0042 contre 0,0040 sur 224 s), dans le bruit (min et max se recouvrent). Modèles `devtrain` / `alldev` : aucun
   effet mesurable.
4. **Conversion m4a** : **0** sur le RTF affiché. `load_audio` (0,10 s sur 13 s, ffmpeg compris) est hors du
   chronomètre de `core.diarize` : `diarize_s` m4a 0,162 s contre wav 0,157 s. Elle compte dans le RTF total (0,0204
   contre 0,0124) et dans « analyse terminée en » (0,3 s contre 0,2 s). Seul effet sur le RTF affiché : la durée
   décodée du m4a est 13,056 s au lieu de 13,0 s (−0,4 % sur le RTF, négligeable).
5. **Premier appel du processus** : RTF 0,027 à 0,031 sur 13 s (0,78 s : chargement des bibliothèques à la première
   utilisation : librosa, STFT, caractéristiques VAD). Hors de l'écart des appels 2+, mais c'est la valeur qu'on voit
   à la **première** analyse après le lancement.

**Le RTF augmente-t-il au 2e lancement ?** Non, pas avec le code actuel : la 1re analyse est la plus lente (0,031
par l'interface), les suivantes sont stables (0,011-0,014) ; pente sur les appels 2-10 non significative (p 0,46 et
0,58). Une hausse d'une analyse à l'autre de 0,006 à 0,013 est dans le bruit observé.

Inexpliqué : la cause exacte de la bimodalité et du surcoût des threads (attente / réveil des threads OpenMP,
hyperthreading sous WSL2 : **hypothèse**, non vérifiée) ; pourquoi 4 threads sont aussi plus rapides sur ldnro
(6,3 s contre 12,3 s, et une forte variance à 12 threads).

pyannote (secondaire) : RTF ≈ 0,26-0,27 au-delà de 2 min, plus bas sur les fichiers courts (le coût suit le nombre
de fenêtres de 10 s, pas la durée). Le m4a de 13 s coûte +31 % (1,53 s contre 1,17 s) : ses 56 ms de plus
(remplissage de l'encodeur AAC) ajoutent une fenêtre ; même temps avec un wav allongé à 13,056 s. 1er appel ≈ 2 à 3 fois
plus lent ; un 1er appel de 11,5 s (RTF 0,88) a été observé une seule fois, au tout premier processus pyannote de la
série (cause non identifiée ; disque froid ou initialisation CUDA : hypothèse).

### Corrections proposées (non appliquées, à décider par Grégoire)

1. **Limiter les threads** pendant l'analyse de la baseline, par ex. `threadpoolctl.threadpool_limits(4)` autour de
   `baseline_adapter.diarize` (déjà installé avec scikit-learn), ou `OMP_NUM_THREADS=4` / `OPENBLAS_NUM_THREADS=4` dans
   `demo/__init__.py`. Gain mesuré : RTF 0,0111 → 0,0040 (13 s), 0,0043 → 0,0035 (224 s), 0,0112 → 0,0058 (ldnro),
   sorties identiques. À ne pas appliquer globalement sans mesurer pyannote sur CPU (plan B), qui utilise aussi des
   threads. La même limite changerait les RTF de `baseline.py` : à décider à part (comparabilité D-36).
2. **Affichage** : la carte « Traitement — lecture du fichier comprise » ne dit pas que la conversion ffmpeg est
   exclue ; afficher la conversion à part (« conversion 0,10 s ») ou préciser « hors conversion ». Indiquer aussi que
   le RTF d'un fichier court (< 30 s) est dominé par un coût fixe.
3. **Fichiers convertis** : `demo/tmp/converted/` garde un wav par analyse (jamais supprimé) ; les supprimer après
   l'analyse ou à l'arrêt.

### Fichiers et reproduction

Créés : `results/demo_agent/rtf_tests/` (`formats*.csv`, `repeat*.csv`, `durations*.csv`, `durations_fit.csv`,
`stages*.csv`, `compare_timing*.csv`, `threads*.csv`, `e2e.csv`, `vadstack_threads.txt`, `scripts/`) ;
`figures/demo/rtf_selon_duree.png` (remplacée). Déplacés : anciens CSV vers `rtf_tests/ancien_premier_agent/`.
Aucune modification de `demo/`, `baseline.py` ni des modules. Scripts dans `rtf_tests/scripts/` (`run_all.sh`
contient les chemins du scratchpad de la session : remplacer `$S` et `$A` par un dossier de travail et les audio
dérivés générés par ffmpeg depuis `data/voxconverse/dev/audio/`). Exemple : `python
results/demo_agent/rtf_tests/scripts/bench.py out.csv tag baseline 10 chemin/afjiv_13s.m4a`.
