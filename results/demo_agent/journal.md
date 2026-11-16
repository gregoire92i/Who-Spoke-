# Journal du subagent `demo-builder` (démo Who Spoke?)

Branche `demo` (créée par Grégoire après le commit 385bc84 « save avant demo »), dépôt principal, aucun commit.
Fichiers audio utilisés : **dev VoxConverse uniquement** (jamais le test), et des fichiers dérivés du dev
(extraits, conversions de format) générés dans `demo/tmp/` ou dans le dossier temporaire de pytest.

## Phase 0 : cartographie (14/11)

- Lecture : `CLAUDE.md`, `docs/etat_du_projet.md`, `docs/choix_techniques.md` (D-05, D-15, D-40, D-50, D-57,
  D-61, D-62), `docs/rapport-ablation.md`, `results/vad_agent/rapport.md`, `README.md`, code des modules.
- `docs/enonce_projet.md` contenait alors une ancienne copie de l'état du projet (corrigé ensuite par Grégoire).
- **Bug observé** : `baseline.py` appelait `best_vad(..., train="traindev")` → `ValueError('traindev')`
  (valeur inconnue, réentraînement tenté). Signalé ; corrigé par Grégoire.
- `librosa.load` et `sf.info` échouent sur `.m4a` (`LibsndfileError`, audioread absent) ; `.mp3` lu.
- pyannote : modèle en cache, chargement hors ligne OK. pqmho (139 s) : GPU 36,9 s (RTF 0,266) puis 34,8 s.
- `best_vad` alldev : 1,04 s au 1er appel sur pqmho (chargement compris), 0,40 s au 2e.

## Phase 1 : propositions et décisions de Grégoire

Décisions reçues (recopiées, non réinterprétées) :
1. Baseline : option B3 — Grégoire ajoute `vad_train="devtrain"` à `baseline.diarize_file` ; la démo appelle
   `vad_train="alldev"`, `frame_size=baseline.frame_size`, `hop_size=baseline.hop_size`. Aucun remplacement de
   fonction à l'exécution. Test d'égalité avec `vad_train="devtrain"` sur 1-2 fichiers dev courts.
2. Modèles `best_vad` manquants : pas d'erreur bloquante ; bandeau d'information + réentraînement si les
   prérequis sont là ; sinon baseline désactivée avec les commandes, pyannote reste disponible.
3. Pipelines : baseline + pyannote (sélecteur), sans variantes.
4. Interface Gradio (6.32.0), timeline plotly. Installation autorisée.
5. pytest autorisé.
6. pyannote : `HF_HUB_OFFLINE=1` fixé avant l'import, sans jeton ; GPU si disponible, sinon CPU (affiché).
7. `.gitignore` : ajout de `demo/tmp/`, `.gradio/`, `*.m4a *.ogg *.opus *.aac *.wma *.webm`.
8. Dépôt principal, pas de worktree ; export RTTM uniquement (pas de JSON).
9. Optionnelles : a (RTTM de référence + DER/JER), b (comparaison côte à côte), c (micro), d (renommage) = oui,
   après les 6 obligatoires ; e (écouter tous les segments d'un locuteur), f (marge d'écoute) = non.

Vérifications phase 1 :
- `pip install --dry-run gradio` : résolution OK sous Python 3.14.4 (roues cp314).
- pyannote hors ligne **sans jeton** (`HF_HUB_OFFLINE=1`, `HF_TOKEN_PATH=/nonexistent/token`) :
  `get_token()` → None, `from_pretrained` 8,7 s, CPU pqmho 131,2 s (RTF 0,944).
- pytest 9.2.0 installé (+ iniconfig 2.3.1, pluggy 1.7.0), ajoutés à `requirements.txt`.

## Phase 2 à 4 : implémentation et tests

- Installation : gradio 6.32.0 + plotly 7.1.0 et 24 dépendances (liste dans le rapport) ; `requirements.txt`
  remplacé par le nouveau `pip freeze` (vérifié : seules 26 lignes ajoutées, aucune retirée ni modifiée).
- `.gitignore` : lignes validées ajoutées.
- API Gradio 6 vérifiée par `inspect.signature` : `Blocks(analytics_enabled=…)`, `launch(server_name, share,
  ssr_mode, mcp_server…)`, `queue(default_concurrency_limit)`, `Audio(sources, type, format, autoplay)`,
  `Dataframe(static_columns, column_count, show_search)`, `DownloadButton`, `SelectData.row_value`.
- API pyannote.core 6.1.0 vérifiée : `Annotation.write_rttm(file)`, `get_overlap()`, `label_timeline(label, copy)`,
  `rename_labels(mapping)`.
- Premier lancement depuis `/tmp` : `ModuleNotFoundError: No module named 'demo'` → la commande doit être lancée
  depuis la racine (documenté). `app.main()` fait ensuite `os.chdir(ROOT)` (chemins relatifs de `vad.py`).
- `gradio_client` : un Dataframe doit être envoyé comme `{"headers": …, "data": …}` (erreur pydantic sinon) ;
  problème du client de test seulement.
- **Problème 1 (corrigé)** : aller-retour RTTM sur une vraie sortie de la baseline → DER 8,43e-05 et non 0.
  Cause : `write_rttm` écrit 3 décimales ; les frontières de la baseline sont des multiples de 32,5 ms (4 décimales) ;
  début et durée sont arrondis séparément (écart ≤ 0,5 ms sur le début, ≤ 1 ms sur la fin). Le test vérifie
  désormais ces tolérances, DER < 1e-3, puis un 2e aller-retour exact (DER = 0).
- **Problème 2 (corrigé)** : pyannote sur un `.mp3` 48 kHz →
  `ValueError: requested chunk [ 00:00:00.000 -->  00:00:10.000] from afjiv file resulted in 478895 samples
  instead of the expected 480000 samples.` Correction dans `demo/core.py` : seuls wav et flac sont passés tels
  quels aux pipelines ; tout autre format est converti en wav 16 kHz mono par ffmpeg (D-50). Test ajouté.

## Cas limites (sortie de `edge_cases.py`, GPU, fichiers dev ou dérivés du dev)

Voir le tableau du rapport (section 6). Observations notables :
- `court_1.5s` : la dernière fin de segment dépasse la durée (1,5275 s pour 1,50 s, baseline ; 1,5497 s pyannote).
  Pour la baseline, cela vient du prolongement des fins de segments de 0,1 s de `best_vad` (`d_end=-0.1`).
- `court_0.3s` : baseline k=0 (aucune window), pyannote k=1 avec 0,4 s de parole (fin 0,402 s > 0,30 s).
- Avertissements numpy « Mean of empty slice » de pyannote sur les fichiers < 2 s (sans effet).
- `abjxc` (1 locuteur en référence) : baseline k=2, pyannote k=1.
- `pqmho` (musique) : baseline 13,1 s de parole, pyannote 15,8 s.
- `.mp3` : pyannote donne 43 segments contre 24 pour le flac du même fichier (décodage différent).
- `ldnro` (1097 s) : baseline 11,05 s (RTF 0,010), pyannote GPU 291,43 s (RTF 0,266), pic RSS 2 854 584 ko.

## Phase 5 : répétition

Voir le rapport (section 7 et 8).

## Embellissement de l'interface (demande de Grégoire, 14/11)

Périmètre : apparence uniquement (`demo/app.py`, `demo/plots.py`, nouveau `demo/style.css`). `core.py`, les
adaptateurs et les pipelines ne changent pas. Aucune dépendance ajoutée, aucune ressource téléchargée.

- Thème `gr.themes.Soft` (hues sky / amber / slate, `radius_size="lg"`), couleur d'accent #0072B2 (1re couleur
  Okabe-Ito), **polices système uniquement** (`font=[ui-sans-serif, system-ui, …]`) : `theme.css` servi ne contient
  que des `--font: ui-sans-serif, system-ui, …` et des `url("data:image/svg+xml…")`, aucune URL externe.
- Dans Gradio 6.32, `theme` et `css_paths` se passent à `launch()` (vérifié par `inspect.signature`).
- Analyse du bundle Gradio : la seule référence à `fonts.googleapis.com` (`Index-CN6InNUr2.js`) est dans la
  fonction d'intégration des Spaces Hugging Face (non appelée en local) ; `cdn.plot.ly` / `unpkg.com` dans
  `PlotlyPlot-*.js` servent aux cartes géographiques / mapbox (non utilisées). Analyse statique, non vérifiée dans un
  navigateur.
- **Test hors ligne** : démo lancée dans un espace de noms réseau sans réseau (`unshare -rn`, loopback seul) :
  « réseau externe : indisponible (attendu) », page servie après 12,88 s, analyse comparée + référence sur afjiv
  (dev), renommage et changement de vue OK par l'API.
- Ressources externes autorisées ensuite par Grégoire (si enregistrées localement avec licence compatible) : **non
  utilisées**, les polices système suffisent.
- Export PNG des figures plotly impossible sans `kaleido` (non installé, pas de nouvelle dépendance) : aperçus HTML
  autonomes (plotly.js embarqué, hors ligne) dans `demo/tmp/preview/` (ignoré par git, 4,8 Mo chacun). Les PNG
  matplotlib de `figures/demo/` ont été régénérés (couleur de la référence #8b93a1).
- Incident évité : un `pkill -f "python -m demo.app"` pour arrêter mon serveur de test a été refusé par le
  système de permissions ; il aurait aussi arrêté la démo de Grégoire (port 7860). Mon serveur (port 7870) a été
  arrêté par son PID (`kill 34802`, après `ss -ltnp` et `ps`).
- `python -m pytest tests/test_demo.py -v` : `16 passed in 6.39s`.

## Point 14 : décisions enregistrées dans `docs/choix_techniques.md` (14/11)

Sur demande explicite de Grégoire (exception du `CLAUDE.md`). Nouvelle section « 8. Démonstration », D-70 à D-76
(numérotation par dizaine et par thème, comme les sections 1 à 7), insérée avant « Décisions ouvertes ». Aucune
entrée existante modifiée ; seule la ligne « Dernière mise à jour » de l'en-tête a été complétée. Les choix propres
de l'agent ne sont pas enregistrés (proposés dans le compte rendu).

## Suppression de la chauffe (décision de Grégoire, appliquée par Claude Code principal, 14/11)

- `demo/pyannote_adapter.py` : plus de `WARMUP_S` ni de passage sur bruit ; `load()` charge seulement la pipeline.
- `demo/baseline_adapter.py` : plus aucun calcul sur un signal au démarrage ; `prepare()` charge les modèles par
  `vad.get_model(train="alldev")` et `vad.get_stack_model(train="alldev")` (réentraînement au démarrage si
  nécessaire) ; `import numpy` retiré.
- Entrée D-77 (chauffe pyannote) retirée de `docs/choix_techniques.md` (par Claude Code principal).
- De mon côté : mentions de la chauffe retirées de `architecture.md` (sections 2 et 4) et de `rapport.md`
  (section 3 et 7 ; les temps de la section 7 sont signalés comme mesurés avec l'ancienne version).

## Tests du RTF (15/11, reprise après un premier agent interrompu)

- Contexte : Grégoire voit un RTF baseline ≈ 0,013 sur un enregistrement de 13 s en m4a, qui augmenterait au 2e
  lancement ; objectif précisé ensuite : expliquer l'écart avec `baseline.py` (médiane 0,0049).
- Vérifié : la chauffe est absente du code (`grep` : plus de `WARMUP`), les mentions restantes de `architecture.md`
  et `rapport.md` disent « sans chauffe » ; tests `16 passed in 6.19s`.
- CSV du premier agent : contenu cohérent mais incomplet (pyannote limité à 5 s pour les durées, 2 appels sur afjiv
  entier ; pas de décomposition ni de test par l'API) et produit avec l'ancienne version → déplacés dans
  `rtf_tests/ancien_premier_agent/`, tout refait. `demo/tmp/rtf_bench/` (audio du premier agent) supprimé.
- Banc : un processus neuf par série, deux pipelines chargées, `load_audio` + `core.diarize` ; ≈ 1 h 10 au total.
  Démarrage à froid observé 21,1 s (smoke test juste après le redémarrage de WSL), 5,6 à 9,6 s ensuite.
- Découverte : `vad_stack` et `compute_features` sont pénalisés par 12 threads OpenMP/OpenBLAS sur un signal court
  (bimodal 0,023-0,094 s ; 0,016 s en 1 thread). `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` : RTF 13 s m4a
  0,0111 → 0,0040, sorties RTTM identiques (`cmp`) sur 3 fichiers. Correction proposée, non appliquée.
- Test par l'API Gradio : serveur `WHO_SPOKE_PORT=7870` (PID 24420), arrêté par `kill 24420` ; le port 7860 était
  libre pendant toute la session. Fichiers créés par ces tests dans `demo/tmp/converted/`, `exports/`, `gradio/`
  supprimés (les fichiers `test_*` antérieurs n'ont pas été touchés).
- Observation : `demo/tmp/converted/` garde un wav par analyse (jamais supprimé) → signalé.
- Hypothèse (non vérifiable ici) : si la 1re analyse de Grégoire affichait 0,013, elle tournait sans doute encore
  avec l'ancienne chauffe ; avec le code actuel la 1re analyse donne ≈ 0,03 sur 13 s.
- Audio dérivés (scratchpad) supprimés en fin de tâche.

## Option embeddings pré-entraînés dans la démo (16/11) : arrêt avant code, options proposées

- Demande : choix MFCC / WeSpeaker (`compute_pretrained_embeddings`) pour la baseline dans la démo.
- Constat : `pretrained_embeddings` est une variable globale de `baseline.py` (l. 38), lue dans `diarize_file`, pas
  un paramètre. Valeur : `True` dans HEAD (c7174d3), `False` dans la copie de travail non committée.
- Seule voie 100 % `demo/` : modifier temporairement `baseline.pretrained_embeddings` (try/finally) → cas prévu
  d'arrêt par la consigne ; aucun code modifié, rapport d'options rendu.
- Vérifications au runtime (scripts dans le scratchpad, supprimés) : modèle
  `pyannote/wespeaker-voxceleb-resnet34-LM` présent dans le cache HF, chargé hors ligne (`HF_HUB_OFFLINE=1`, `token=False`) ;
  `torch.cuda.is_available()` = True ; chargement 3,1 s (CPU, 1er) / 0,36 s (GPU) ; 598 windows de 2 s sur signal
  aléatoire de 10 min : CPU 36,87 s puis 36,25 s ; GPU 9,68 s puis 8,94 s (GPU en mode idle, non représentatif, D-36).

### Suite : implémentation (16/11, après décision de Grégoire)

- Décision reçue (via le coordinateur) : variante de l'option B. `baseline.diarize_file(..., pretrained_embeddings=False)`
  modifié par le coordinateur (pas par moi) ; défaut qui fait foi pour la démo : **False (MFCC)**.
- `demo/baseline_adapter.py` : `DEFAULT_PRETRAINED = False`, `D40_NOTE`, `prepare_wespeaker()` (préchargement au
  démarrage, `load_pretrained_model(WESPEAKER_ID, None)`, hors ligne ; en cas d'échec la baseline MFCC reste
  disponible), `params(pretrained_embeddings)` avec la clé `embeddings`, `diarize(..., pretrained_embeddings=)` passé
  explicitement à `baseline.diarize_file` ; `extra["pretrained_embeddings"]`.
- `demo/core.py` : `Engines.wespeaker` (état du préchargement) ; `diarize(audio, pipeline, engines,
  pretrained_embeddings=False)` refuse WeSpeaker s'il n'est pas chargé (RuntimeError, pas de bascule silencieuse).
- `demo/app.py` : radio « Embeddings de la baseline » (MFCC par défaut ; désactivée si WeSpeaker indisponible),
  libellés dynamiques (`_label`, `_short`) dans le titre, le sélecteur de vue, la timeline, le tableau DER et le
  bouton RTTM ; fichier exporté `<uri>_baseline_wespeaker.rttm` ; note D-40 sous les résultats WeSpeaker.
  Le contenu du RTTM ne change pas (format standard, l'URI ne contient pas le pipeline).
- Tests : 16 → 22 (`22 passed in 14.34s`) ; égalité adaptateur = `baseline.diarize_file` dans les deux modes sur
  hqyok et tucrg (dev, vad_train="devtrain").
- Vérification manuelle hqyok (alldev, GPU) : MFCC k=2 RTF 0,02 ; WeSpeaker k=3 RTF 0,0535. Les RTTM d'export
  `demo/tmp/exports/hqyok_baseline*.rttm` produits par cette vérification ont été supprimés.
