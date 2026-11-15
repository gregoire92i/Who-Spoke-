# Architecture de la démo Who Spoke?

> Rédigé par Claude Code (subagent `demo-builder`) le 14/11/2026, après l'embellissement de l'interface.
> Complète `results/demo_agent/rapport.md` (résultats, tests, temps, risques) sans le remplacer.
> Décrit le code tel qu'il existe dans `demo/` sur la branche `demo` (non committé à cette date).

## 1. Vue d'ensemble

L'application est une page web **locale** (Gradio, `http://127.0.0.1:7860`). On y charge un fichier audio (ou un
enregistrement micro), puis on choisit une pipeline : la **baseline** du projet ou **pyannote**
`speaker-diarization-community-1`. L'application calcule la diarisation, c'est-à-dire une `pyannote.core.Annotation`,
puis affiche :
- la timeline des locuteurs ;
- leurs temps de parole ;
- le tableau des segments, qu'on peut écouter.

Elle permet d'exporter le résultat en **RTTM**. Elle propose aussi quatre options :
- a. une référence RTTM, pour calculer DER et JER sur ce fichier ;
- b. la comparaison des deux pipelines ;
- c. l'enregistrement au micro ;
- d. le renommage des locuteurs.

```
                     demo/app.py  (interface Gradio : mise en page, événements, état de session)
                        │   utilise
          ┌─────────────┼─────────────────────┐
          ▼             ▼                     ▼
   demo/core.py     demo/plots.py        demo/style.css
   (cœur, sans      (figures plotly       (apparence)
    interface)       + PNG matplotlib)
      │      │
      │      ├──► demo/baseline_adapter.py ──► baseline.py : diarize_file(..., vad_train="alldev")
      │      │                                   ├─ audio_features.py (load_audio, FFT, compute_mfcc)
      │      │                                   ├─ vad.py (best_vad → modèles results/vad_agent/models/*alldev*)
      │      │                                   ├─ embeddings.py, clustering.py (NME-SC), post_treatment.py
      │      │                                   └─ constantes de baseline.py (frame_size, hop_size, …)
      │      └──► demo/pyannote_adapter.py ──► pyannote_diarization.py : load_pipeline, diarize_file
      │                                         └─ modèle en cache ~/.cache/huggingface/hub (hors ligne)
      └──► pyannote.core / pyannote.database / pyannote.metrics (Annotation, load_rttm, DER, JER)

   demo/__init__.py : exécuté en premier ; fixe les variables d'environnement ; ROOT, TMP_DIR
   demo/tmp/        : fichiers temporaires (ignoré par git)
```

**Principe directeur** : `demo/` ne définit **aucun paramètre de pipeline**. La baseline appelle
`baseline.diarize_file` avec les constantes de `baseline.py`. Seul `vad_train="alldev"` est imposé (modèles de
production de `best_vad`). pyannote est appelé par les fonctions de `pyannote_diarization.py`.

## 2. Rôle de chaque fichier de `demo/`

### `demo/__init__.py`
Il est exécuté avant tout autre module du paquet, car `python -m demo.app` importe d'abord `demo`.
- Il fixe `HF_HUB_OFFLINE=1`, `GRADIO_ANALYTICS_ENABLED=False`, `GRADIO_TEMP_DIR=demo/tmp/gradio` (par défaut) et
  `MPLBACKEND=Agg` (par défaut).
- `ROOT` est la racine du dépôt, `TMP_DIR` vaut `ROOT/demo/tmp`.

### `demo/core.py` : cœur indépendant de l'interface

| Nom | Entrées | Sortie | Rôle |
|---|---|---|---|
| `SR`, `PIPELINES`, `AUDIO_EXTENSIONS`, `DIRECT_FORMATS` | — | constantes | 16000 Hz ; `("baseline", "pyannote")` ; extensions proposées ; formats passés tels quels (`WAV`, `WAVEX`, `FLAC`) |
| `AudioLoadError` | — | exception | fichier absent, vide, corrompu ou illisible |
| `make_uri(path)` | chemin | `str` | nom sans extension, accents retirés, caractères hors `[A-Za-z0-9_-]` remplacés par `_` (« audio » si vide) |
| `_ffmpeg_to_wav(path, uri)` | chemin, uri | chemin du wav | conversion ffmpeg en wav 16 kHz mono PCM 16 bits dans `demo/tmp/converted/` |
| `load_audio(path)` | chemin | `dict` : `y`, `sr`, `duration`, `pipeline_path`, `uri`, `converted`, `original_sr`, `original_channels` | lit l'audio (librosa, 16 kHz mono) et choisit le fichier donné aux pipelines |
| `Engines(load_pyannote=True, load_baseline=True, log=print)` | — | objet : `status`, `pyannote`, `device`, `available()` | charge les modèles **une fois** ; `status[p] = (disponible, message)` |
| `diarize(audio, pipeline, engines)` | sortie de `load_audio`, `"baseline"` ou `"pyannote"`, `Engines` | `(Annotation, infos)` | point d'entrée unique ; `infos` : pipeline, params, `processing_time_s`, `duration_s`, `RTF`, `k_hyp` (+ `vad_speech_s`, `p_neighbors_used`, `n_windows` pour la baseline) |
| `speaker_stats(annotation, duration)` | Annotation, durée | `(lignes, résumé)` | temps par locuteur (s, % de la parole, % de l'audio, nb de segments) ; résumé : parole, overlap, durée, nb de locuteurs |
| `segments_table(annotation)` | Annotation | liste `(start, end, duration, label)` | segments triés par début |
| `segment_audio(y, sr, segment, margin=0.0)` | signal, sr, `Segment` ou `(start, end)` | sous-signal | extraction pour l'écoute, bornes coupées à `[0, len(y)]` |
| `rename(annotation, names)` | Annotation, `{ancien: nouveau}` | nouvelle Annotation | renommage ; espaces → `_` ; doublons → `ValueError` |
| `to_rttm(annotation, uri, path)` | Annotation, uri, chemin | chemin | `Annotation.write_rttm` ; annotation vide → fichier vide |
| `load_reference(path, uri)` | RTTM, uri | Annotation | `load_rttm` ; prend l'uri du fichier, ou la seule présente |
| `evaluate(reference, hypothesis, duration)` | 2 Annotations, durée | `dict` : DER, DER_forgiving, JER, false_alarm, missed, confusion, ref_speech_s, n_ref_speakers | protocole D-30 à D-37 (collar 0, UEM [0, durée]) |
| `fmt_time(t)` | secondes | `"m:ss.s"` | affichage |

### `demo/baseline_adapter.py`

| Nom | Rôle |
|---|---|
| `VAD_TRAIN = "alldev"`, `MODELS_DIR`, `MODEL_FILES`, `RETRAIN_CMD` | modèles de production de `best_vad` : `hgb_base+local_s4_trainalldev.pkl`, `stack_hgb_base+local_s4_alldev.pkl` |
| `check_vad_models()` | renvoie `("ok" \| "retrain" \| "missing", message)` selon la présence des modèles et des prérequis du réentraînement |
| `prepare(log)` | vérifie, puis charge les 2 modèles par `vad.get_model(train="alldev")` et `vad.get_stack_model(train="alldev")`, sans calcul sur un signal (réentraînement au démarrage si nécessaire) ; renvoie `(disponible, message)` |
| `params()` | paramètres lus dans `baseline.py` (`frame_ms`, `hop_ms`, `n_mfcc`, `win_s`…, `clustering_method`, `nme_*`, `min/max_speakers`, `seed`) et `vad.PROD_PARAMS` (affichage) |
| `diarize(path, uri, vad_train=VAD_TRAIN)` | `baseline.diarize_file(Path(path), uri, frame_size=b.frame_size, hop_size=b.hop_size, vad_train=vad_train)` → `(hyp, params, extra)` |

### `demo/pyannote_adapter.py`

| Nom | Rôle |
|---|---|
| `load(log)` | appareil : `cuda` si disponible, sauf si `WHO_SPOKE_DEVICE=cpu` ; `pyannote_diarization.load_pipeline(device)`, sans chauffe ; renvoie `(pipeline, device)` |
| `params()` | `model_id`, version de pyannote.audio, nombre de locuteurs estimé (`oracle=False`) |
| `diarize(pipeline, path)` | `pyannote_diarization.diarize_file(pipeline, path)` → `(hyp, params, {})` ; `hyp` est `speaker_diarization`, overlap compris |

### `demo/plots.py`

| Nom | Rôle |
|---|---|
| `OKABE_ITO`, `PATTERNS`, `OVERLAP_COLOR`, `REF_COLOR`, `FONT`, `GRID` | palette, hachures, couleurs, police système, grille |
| `speaker_order(annotation)` | locuteurs triés par temps de parole décroissant |
| `style(i)`, `colors_for(annotation)` | `(couleur, hachure)` du i-ème locuteur ; dict `label → (couleur, hachure)` |
| `_style_layout(fig, title, **kw)` | style commun plotly : fond transparent, gris #8b93a1, grille discrète |
| `timeline_figure(annotation, duration, title, reference=None, row_prefix="")` | timeline plotly : une ligne par locuteur, ligne `overlap`, lignes `réf.` optionnelles, zoom, rangeslider si > 120 s |
| `bars_figure(annotation, stats_rows)` | barres horizontales du temps de parole, mêmes couleurs |
| `save_timeline_png(annotation, duration, path, title, reference=None)` | même timeline en matplotlib (Agg) pour `figures/demo/` |

### `demo/app.py`

| Nom | Rôle |
|---|---|
| `make_theme()` | thème `gr.themes.Soft`, polices système, accent #0072B2 |
| `_header`, `_section`, `_card`, `_legend`, `_info_html`, `_metrics_html` | fragments HTML (en-tête, titres de section, cartes de chiffres, légende, tableau DER / JER) |
| `_styled`, `_stats_df`, `_seg_df`, `_text_color` | tableaux pandas avec la colonne « locuteur » colorée (`Styler.map`) |
| `_current(state, view)`, `_view_outputs(state, view)`, `_timelines(state)` | annotation renommée, ensemble des sorties d'une vue, figure(s) de timeline |
| `build_app(engines)` | construit les composants et branche les événements (`analyse`, `change_view`, `play`, `apply_names`) |
| `main()` | `os.chdir(ROOT)`, `Engines()`, `build_app`, `launch(...)` |

### `demo/style.css`
Feuille de style, voir la section 6.

## 3. Parcours d'une analyse de bout en bout

| # | Étape | Fonction (fichier) | Données |
|---|---|---|---|
| 1 | Upload | composant `gr.File` (`app.py`) | Gradio copie le fichier dans `demo/tmp/gradio/<hash>/` et passe son chemin à `analyse` |
| 2 | Choix de l'entrée | `analyse` (`app.py`) | fichier sinon micro (`mic_path`) ; erreur Gradio si aucun |
| 3 | Lecture, conversion | `core.load_audio` → `_ffmpeg_to_wav` si besoin (`core.py`) | `sf.info` : wav / flac → fichier d'origine ; autre format ou illisible par soundfile → wav 16 kHz mono dans `demo/tmp/converted/` ; échec → `AudioLoadError` → message |
| 4 | Signal pour l'écoute | `librosa.load(..., sr=16000, mono=True)` (`core.load_audio`) | `y` (float32), `duration = len(y)/16000`, `uri` |
| 5 | Diarisation | `core.diarize` → `baseline_adapter.diarize` ou `pyannote_adapter.diarize` | la pipeline relit `pipeline_path` ; chronomètre autour de l'appel → `processing_time_s`, `RTF` ; `ann.uri = uri` nettoyée |
| 6 | (option a) Référence | `core.load_reference` | Annotation de référence, ou avertissement si illisible |
| 7 | Statistiques | `core.speaker_stats` (via `_view_outputs`) | lignes par locuteur + résumé |
| 8 | Affichage | `plots.timeline_figure`, `plots.bars_figure`, `_info_html`, `_stats_df`, `_seg_df`, `_metrics_html` (`core.evaluate`) | figures plotly, HTML des cartes, tableaux colorés |
| 9 | Écoute | `play` (`app.py`) → `core.segment_audio` | `SelectData.row_value[0:2]` (début, fin en s) → `(16000, int16)` vers `gr.Audio` |
| 10 | Export RTTM | `core.to_rttm` (via `_view_outputs`) | `demo/tmp/exports/<uri>_<pipeline>.rttm` (labels renommés s'il y a lieu) → `gr.DownloadButton` |

Le RTTM est **réécrit à chaque affichage** d'une vue (analyse, changement de vue, renommage). Le fichier téléchargé
correspond donc toujours à ce qui est affiché.

## 4. Démarrage

Commande, depuis la racine : `python -m demo.app`.
1. Import de `demo/__init__.py` : variables d'environnement. Elles sont fixées **avant** l'import de
   huggingface_hub, pyannote et gradio.
2. Import de `gradio`, `core`, `plots`.
3. `main()` : `os.chdir(ROOT)`, nécessaire parce que `vad.py` lit ses modèles par des chemins relatifs. Création de
   `demo/tmp/gradio`.
4. `core.Engines()` :
   - **baseline d'abord** : `baseline_adapter.prepare` → `check_vad_models` → `vad.get_model` et `vad.get_stack_model`
     (`train="alldev"`). Les modèles restent en mémoire dans `vad._MODELS` ; aucun calcul sur un signal ;
   - **puis pyannote** : `pyannote_adapter.load` → `load_pipeline(device)` (hors ligne, sans jeton), sans chauffe.
     Une exception désactive pyannote sans arrêter la démo.
5. `build_app(engines)`, puis
   `queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=…, share=False, ssr_mode=False, mcp_server=False, theme=make_theme(), css_paths=[style.css])`.

Durées observées sur ce PC (GPU) :
- modèles prêts en 7,4 à 9,2 s ;
- page servie après 11,25 à 12,9 s ;
- premier lancement à froid : 23,2 s pour `Engines`.

| Variable | Effet |
|---|---|
| `HF_HUB_OFFLINE=1` | forcée : pyannote lu dans le cache local, sans réseau ni jeton |
| `WHO_SPOKE_DEVICE=cpu` | force pyannote sur CPU (plan B) |
| `WHO_SPOKE_PORT` | port (7860 par défaut) |
| `GRADIO_ANALYTICS_ENABLED=False` | forcée : pas de statistiques envoyées |
| `GRADIO_TEMP_DIR` | `demo/tmp/gradio` par défaut (uploads) |
| `MPLBACKEND=Agg` | pas d'affichage matplotlib |

## 5. Interface Gradio

**Mise en page** (de haut en bas) :
- en-tête (`gr.HTML`) : titre, sous-titre, pastilles d'état des pipelines ;
- colonne gauche :
  - section 1 « Audio » : `gr.File`, accordéon micro (`gr.Audio(sources=["microphone"])`) ;
  - section 2 « Analyse » : `gr.Radio` pipeline, `gr.Checkbox` « Comparer », accordéon RTTM de référence ;
  - bouton « Lancer l'analyse » ;
- colonne droite :
  - ligne d'état du fichier ;
  - `gr.Radio` « Résultats affichés » (visible seulement en comparaison) ;
  - cartes de chiffres clés, légende et paramètres repliés ;
  - tableau des métriques (option a) ;
- section 3 : timeline (`gr.Plot`), plus une seconde timeline en comparaison ;
- section 4 : tableau du temps de parole + barres ;
- section 5 : tableau des segments, lecteur du segment, bouton de téléchargement RTTM, accordéon de renommage.

**Événements :**

| Déclencheur | Fonction | Sorties |
|---|---|---|
| bouton « Lancer l'analyse » (`run.click`) | `analyse(file, mic, pipeline, compare, ref)` | état, ligne d'état, timelines, sélecteur de vue, métriques, sorties de vue |
| sélecteur « Résultats affichés » (`view.change`) | `change_view(state, view)` | sorties de vue (cartes, tableaux, barres, RTTM…) |
| clic sur une ligne du tableau des segments (`seg_df.select`) | `play(state, view, evt)` | lecteur du segment |
| bouton « Appliquer les noms » (`names_btn.click`) | `apply_names(state, view, names)` | état, timelines, sorties de vue |

**État conservé entre les actions** (`gr.State`, propre à chaque onglet du navigateur) :
- `audio` : sortie de `load_audio`, signal compris ;
- `results` : `{pipeline: (Annotation, infos)}` ;
- `names` : `{pipeline: {ancien: nouveau}}` ;
- `ref` : Annotation de référence ou `None`.

L'annotation d'origine n'est jamais modifiée : le renommage est appliqué à l'affichage par `core.rename`. Une seule
analyse tourne à la fois (`default_concurrency_limit=1`).

**Options :**
- **a. Référence** : le RTTM est lu par `core.load_reference`. Ses locuteurs sont ajoutés en lignes « réf. » grises
  sur les timelines. `core.evaluate` donne DER, DER forgiving, JER, FA, missed et confusion pour chaque pipeline
  calculée, avec la mention « ce fichier uniquement ».
- **b. Comparaison** : `analyse` lance toutes les pipelines disponibles. Deux timelines s'affichent, et le sélecteur
  « Résultats affichés » choisit celle dont on voit les cartes, tableaux, écoute et RTTM.
- **c. Micro** : le fichier wav enregistré par le navigateur est utilisé si aucun fichier n'est chargé.
  L'uri vaut `micro_AAAAMMJJ_HHMMSS`.
- **d. Renommage** : tableau `locuteur | nouveau nom`. Les noms vides sont ignorés, les doublons refusés. Le
  renommage s'applique à la timeline, aux tableaux, aux barres et au RTTM.

## 6. Style

- **Thème** : `gr.themes.Soft(primary_hue="sky", secondary_hue="amber", neutral_hue="slate", radius_size="lg")`, avec :
  - accent et bouton principal #0072B2 (#2a8fc7 en mode sombre) ;
  - **polices système** : `ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif`
    et une pile monospace système.
- **CSS** (`demo/style.css`) :
  - en-tête en dégradé bleu ;
  - titres de section numérotés ;
  - cartes `.ws-card` ;
  - pastilles de locuteurs `.ws-chip` ;
  - avertissement « aucune parole » ;
  - tableau des métriques.

  Les couleurs du texte et des fonds passent par les variables CSS de Gradio (`--body-text-color`,
  `--block-background-fill`, `--block-border-color`, `--color-accent`) pour suivre le mode clair ou sombre.
- **Figures plotly** :
  - fond transparent ;
  - texte gris moyen #8b93a1, lisible sur fond clair comme sur fond sombre ;
  - grille discrète ;
  - segments aux coins arrondis.
- **Palette des locuteurs** : Okabe-Ito (lisible par les daltoniens), dans l'ordre du temps de parole décroissant :
  #0072B2, #E69F00, #009E73, #CC79A7, #56B4E9, #D55E00, #F0E442, #000000. Au-delà de 8 locuteurs, les couleurs
  reviennent avec une hachure.
  - La même fonction `colors_for` sert à la timeline, aux barres, à la légende et à la colonne « locuteur » des deux
    tableaux (texte noir ou blanc selon la luminance).
  - Overlap : gris #7f7f7f hachuré. Référence : #8b93a1.
- **Ressources externes** : **aucune**. Aucune police ni icône téléchargée, donc aucune licence tierce à déclarer
  pour `demo/`.

## 7. Erreurs et cas limites

| Situation | Comportement |
|---|---|
| Modèles `best_vad` présents | état `ok` |
| Un modèle absent, prérequis présents (cache `results/vad_agent/cache/dev/*.npz` complet, `samples/dev/all_dev.rttm`, annotations `data/voxconverse/annotations/dev/*.rttm`) | état `retrain` : message « réentraînement automatique en cours » dans le terminal pendant le démarrage ; `vad.py` réentraîne seul (≈ 3-5 min) ; l'en-tête affiche ensuite « baseline prête (réentraînement terminé) ». Chemin simulé dans les tests, réentraînement réel non exécuté |
| Un modèle absent, prérequis absents | état `missing` : baseline retirée du sélecteur, message avec `RETRAIN_CMD` ; pyannote reste disponible |
| pyannote impossible à charger (cache absent…) | pastille rouge, pyannote absent du sélecteur |
| Aucune pipeline disponible | bouton « Lancer l'analyse » désactivé |
| VAD vide (silence, bruit, audio très court) | Annotation vide : carte « Aucune parole détectée », timeline vide, RTTM vide (valide) |
| Format illisible, fichier corrompu ou vide | `AudioLoadError` → `gr.Error` (message, pas de crash) |
| Format non wav / flac | conversion ffmpeg ; nécessaire pour pyannote, qui échoue sur la lecture directe d'un mp3 |
| Pas de GPU | pyannote sur CPU, appareil affiché dans les cartes ; RTF observé 0,946 |
| Erreur pendant la diarisation | `gr.Error("Échec de <pipeline> : …")` |
| RTTM de référence illisible | `gr.Warning`, analyse conservée sans métriques |

## 8. Confidentialité (D-05)

- Le serveur n'écoute que sur `127.0.0.1`, sans `share` ni SSR ni serveur MCP. Les analytics Gradio sont coupées.
  pyannote est hors ligne et aucun jeton n'est lu par la démo.
- Fichiers créés :
  - `demo/tmp/gradio/` : uploads ;
  - `demo/tmp/converted/` : wav convertis ;
  - `demo/tmp/exports/` : RTTM.

  Tout est sous `demo/tmp/`, ignoré par git (`.gitignore` : `demo/tmp/`, `.gradio/`, `*.m4a`, `*.ogg`, `*.opus`,
  `*.aac`, `*.wma`, `*.webm`, en plus de `*.wav`, `*.mp3`, `*.flac`, `data/`). Ces fichiers ne sont pas supprimés
  automatiquement : les effacer après la soutenance.
- **Vérification hors ligne** : démo lancée dans un espace de noms réseau sans accès externe (`unshare -rn`). La page
  a été servie et une analyse complète a été faite par l'API.
  - Le `theme.css` servi ne contient aucune URL externe.
  - Le bundle Gradio contient bien des URLs externes, mais dans du code non utilisé ici : intégration des Spaces
    Hugging Face, cartes plotly géographiques.
  - **Non vérifié dans un navigateur** (onglet « Réseau » des outils de développement).

## 9. Tests (`tests/test_demo.py`, 16 tests)

| Test | Vérifie |
|---|---|
| `test_rttm_roundtrip_synthetic` | écriture puis relecture `load_rttm` : DER = 0 (moderne et forgiving) ; 10 champs par ligne |
| `test_rttm_roundtrip_pipeline_output` | sur une sortie réelle de la baseline (afjiv) : mêmes labels, écarts ≤ 0,5 ms (débuts) et ≤ 1 ms (fins), DER < 1e-3 ; second aller-retour exact |
| `test_rttm_empty_annotation` | annotation vide → fichier vide, relu comme `{}` |
| `test_make_uri` | espaces, parenthèses, accents, caractères non latins |
| `test_speaker_stats_consistent` | temps par locuteur, parole, overlap, somme = parole + overlap, tri, pourcentages |
| `test_speaker_stats_empty` | annotation vide |
| `test_segment_audio_bounds` | bornes en échantillons, fin de fichier, marge, segment hors fichier |
| `test_load_audio_formats` | wav ; flac stéréo 44,1 kHz avec espaces ; m4a converti dans `demo/tmp/` ; mp3 48 kHz converti en wav 16 kHz mono |
| `test_load_audio_errors` | fichier corrompu, texte en `.mp3`, fichier vide, fichier absent → `AudioLoadError` |
| `test_adapter_equals_baseline_devtrain[hqyok]`, `[tucrg]` | adaptateur = `baseline.diarize_file` avec `vad_train="devtrain"` |
| `test_demo_uses_alldev` | `core.diarize` = `baseline.diarize_file(..., vad_train="alldev")` ; params et RTF |
| `test_vad_models_present` | `check_vad_models()` vaut `ok` |
| `test_evaluate_reference_against_itself` | DER = DER forgiving = JER = 0 |
| `test_rename` | renommage, original intact, doublons refusés |
| `test_vad_models_missing_states` | modèles absents simulés : `retrain` puis `missing`, baseline indisponible, `diarize` lève `RuntimeError` |

Commande : `python -m pytest tests/test_demo.py -v`. Les tests qui lisent l'audio dev sont sautés si
`data/voxconverse/dev/` est absent. Aucun test n'utilise le test VoxConverse.

## 10. Modifier ou étendre

- **Changer un paramètre de la baseline** : le modifier dans `baseline.py`, où les valeurs font foi. La démo le
  reprend sans changement. Le test d'égalité vérifie que l'adaptateur suit.
- **Changer la VAD de la démo** : `VAD_TRAIN` dans `demo/baseline_adapter.py`. Si le nom des modèles change, mettre
  aussi à jour `MODEL_FILES`.
- **Ajouter une pipeline** :
  1. créer `demo/<nom>_adapter.py` avec `load` (si un modèle est à charger) et
     `diarize(...) -> (Annotation, params, extra)` ;
  2. l'ajouter à `core.PIPELINES` ;
  3. ajouter le chargement dans `Engines.__init__` et la branche correspondante dans `core.diarize` ;
  4. ajouter son libellé dans `LABELS` et `SHORT` (`app.py`).

  L'option « Comparer » lance toutes les pipelines disponibles.
- **Ajouter un format** :
  - tout ce que ffmpeg lit est déjà converti. Il suffit d'ajouter l'extension à `core.AUDIO_EXTENSIONS`, pour le
    filtre du sélecteur de fichier ;
  - pour passer un format sans conversion, l'ajouter à `DIRECT_FORMATS` (nom de format soundfile), après avoir
    vérifié que pyannote le lit.
- **Style** : `demo/style.css` (classes `ws-*`) et `make_theme()` dans `app.py`. Pour les figures : `FONT`, `GRID` et
  `_style_layout` dans `plots.py`.

## 11. Limites connues et points non vérifiés

- **Rendu dans un navigateur jamais vu par l'agent** (pas de navigateur) :
  - mise en page, mode sombre, couleurs des tableaux ;
  - clic sur un segment, lecture audio ;
  - barre de progression, micro ;
  - absence de requêtes réseau côté navigateur.

  Tout a été vérifié par l'API (`gradio_client`) et par analyse du HTML et du CSS servis.
- **Aperçus des figures** : pas d'export PNG des figures plotly (il faudrait `kaleido`, non installé). Aperçus HTML
  autonomes dans `demo/tmp/preview/`, non versionnés. PNG matplotlib équivalents dans `figures/demo/`.
- **Couleurs** : les couleurs des locuteurs dépendent de l'ordre de temps de parole. Elles peuvent donc différer
  entre deux pipelines pour un même locuteur réel.
- **Lecture des segments** : elle utilise les bornes arrondies à la ms du tableau.
- **Contraintes d'exécution** : la commande doit être lancée depuis la racine du dépôt. `demo/tmp/` n'est pas vidé
  automatiquement.
- **Exactitude des sorties** :
  - les fins de segment peuvent dépasser légèrement la durée du fichier (jusqu'à environ 0,05 s observé), à cause du
    prolongement de 0,1 s de `best_vad` et du comportement de pyannote ;
  - le RTTM arrondit à la ms.
- **Résultats de pyannote** : ils diffèrent selon le format d'origine (mp3 converti : 43 segments contre 24 pour le
  flac du même fichier).
- **RTF de pyannote** : le RTF GPU n'est pas représentatif sur ce PC.
