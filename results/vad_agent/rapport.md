# Rapport : VAD de Who Spoke (VoxConverse), version conservée

Ce rapport ne décrit que la version conservée. L'historique complet se trouve dans `journal.md` : 21 itérations, hypothèses
réfutées, coûts, versions écartées et nettoyage.

## Protocole
- **Critère** : detection error = (FA + MISS) / durée de parole de référence. Calcul avec
  `DetectionErrorRate(collar=0.0, skip_overlap=False)`, UEM = [0, durée du wav], agrégation micro (identique à `main()`).
- **Référence** : `energy_vad` d'`audio_features.py`, importée et non modifiée, avec les paramètres de `baseline.py` :
  - trame 65 ms, hop 32,5 ms ;
  - seuil -30 dB, delta 3,18, lissage 3 trames ;
  - min_silence 0,5 s, min_speech 0,25 s.
- **Données** :
  - apprentissage sur les 126 fichiers dev hors sample90 (« dev-train ») ;
  - réglage sur dev sample90 (90 fichiers, 26658,9 s de parole) ;
  - validation sur test sample90 (57473,3 s de parole), sans aucun réglage.
- **Objectif fixé** : < 3 % sur le réglage. **Non atteint (3,131 %)**.

## 1. VAD conservée : `vad_stack` (itération 12)
1. **Premier étage** : HistGradientBoosting (300 itérations, 63 feuilles) par trame de 65 ms / 32,5 ms, avec les
   caractéristiques « base+local » (118 dimensions) :
   - log-mel 40 bandes normalisé par fichier ;
   - énergie relative au p95 et au p5 du fichier ;
   - platitude, entropie 250-4000 Hz, centroïde, ZCR, périodicité (autocorrélation) ;
   - moyenne et écart-type sur 11 et 31 trames, variabilité spectrale ;
   - creux d'énergie locaux (énergie totale et 300-3400 Hz relatives au max et à la moyenne glissants, pentes).
2. **Second étage** : HGB sur le contexte temporel des log-odds du premier étage (décalages ±20 trames, min, max et moyenne
   sur 5, 11 et 31 trames), plus les caractéristiques du premier étage. À l'entraînement, les log-odds sont prédits
   hors-échantillon (3 plis par fichier).
3. **Décision** : log-odds > 0,5, puis comblement des trous < 0,5 s, suppression de la parole < 0,25 s, et fins de segments
   prolongées de 0,1 s.

Deux jeux de modèles :
- **Évaluation** : entraînée sur dev-train. Commande `python vad.py eval --method best` (`BEST_PARAMS`).
- **Production** : entraînée sur les 216 fichiers dev, fonction `best_vad(y)` (`PROD_PARAMS`). Elle n'est pas évaluable sur
  sample90, qui fait partie de son entraînement.

## 2. Résultats

| Jeu | VAD | FA | Miss | Critère |
|---|---|---|---|---|
| Réglage (dev sample90) | `energy_vad` (référence) | 4.177 % (1113.4 s) | 2.472 % (659.0 s) | 6.649 % |
| Réglage | `vad_stack`, dev-train | 2.003 % (533.9 s) | 1.129 % (300.9 s) | **3.131 %** |
| Test sample90 | `energy_vad` (référence) | 7.658 % (4401.4 s) | 3.387 % (1946.7 s) | 11.045 % |
| Test | `vad_stack`, dev-train | 3.371 % (1937.3 s) | 2.390 % (1373.5 s) | **5.761 %** |
| Test | `vad_stack`, all-dev (production) | 3.459 % (1987.7 s) | 1.976 % (1135.5 s) | **5.434 %** |

- **Gain sur la référence** :
  - réglage : -3.518 pt (-52,9 %) ;
  - test : -5.284 pt (-47,8 %) avec dev-train, -5.611 pt (-50,8 %) avec all-dev.
- **Par fichier sur le test** :
  - la version dev-train améliore 81 fichiers sur 90 par rapport à la référence ;
  - la version all-dev en améliore 52 et en dégrade 37 par rapport à la version dev-train.
- **Coût** :
  - entraînement : environ 1 min 45 s pour dev-train (3 plis + 2e étage) et 203 s pour all-dev ;
  - inférence : 2 modèles HGB ;
  - pour l'évaluation, un cache de caractéristiques d'environ 10 min pour le dev (3 processus).

## 3. Changements dans `vad.py`
`git diff --stat -- vad.py` : +446 / -1 (651 lignes contre 205 à l'origine). La seule ligne retirée est l'appel `main()`, remplacé par
`cli()`, qui appelle encore `main()` sans argument. La partie d'origine (`main`, `trace`, `hysteresis`, `energy_vad` locale) est
inchangée.

Ajouts :

| Bloc | Rôle |
|---|---|
| Banc d'évaluation : `evaluate`, `fmt`, `_parse_value`, `cli` | CLI `cache` et `eval`, protocole de `main()` |
| `compute_features`, `periodicity`, `build_cache`, `load_cache`, `load_ref`, `frame_labels` | caractéristiques et vérité terrain mises en cache |
| `vad_reference` (sortie d'`energy_vad` mise en cache), `vad_energy` | référence, et ré-implémentation de contrôle qui la reproduit exactement |
| `build_X` (base, base+local), `training_set`, `train_items`, `get_model`, `speech_llr` | premier étage |
| `stack_X`, `oof_llr`, `get_stack_model`, `vad_stack`, `adjust_bounds`, `frames_to_timeline` | second étage et post-traitement |
| `BEST_PARAMS`, `PROD_PARAMS`, `best_vad` | configuration retenue et point d'entrée de production |

Origine des gains, sur le réglage (détail dans le journal) :

| Itérations | Changement | Critère |
|---|---|---|
| 0 | référence | 6.649 % |
| 3 | classifieur HGB | 3.531 % |
| 7 | creux locaux | 3.392 % |
| 10 | fins de segments + 0,1 s | 3.296 % |
| 12 | second étage | 3.131 % |

## 4. Limites
- **Plateau.** Environ 0.7 pt d'erreur se trouve à ±50 ms des frontières de référence, et ne diminue pas avec une décision à
  10 ms. Le comblement des pauses < 0,5 s coûte 0.94 pt même avec un oracle. HMM, augmentation et ensemble n'ont rien apporté
  (voir le journal, itérations 13 à 20).
- **Pires fichiers test (production)** : wprog (FA 45.7 s, Miss 170.9 s), wemos (58.1 / 61.2), uicid (Miss 95.1),
  diysk (FA 99.4), bgvvt (64.1 / 31.9). Sans ces 5 fichiers, l'erreur test est de 4.671 %. Je suppose de la parole sur fond
  musical et des jingles, mais je n'ai écouté aucun fichier.
- **Seuil non réajusté pour all-dev.** Le seuil 0,5 a été réglé pour les modèles dev-train.
- **Données externes.** Pour traiter la parole sur musique, il faudrait un corpus externe de musique et de bruit (MUSAN par
  exemple). Je ne l'ai pas téléchargé.
- **Test consulté 4 fois** au cours du projet. La version évaluée était à chaque fois fixée avant l'exécution, et aucun choix
  n'a été fait d'après le test.

## 5. Reproduction (depuis la racine du dépôt)
```
python vad.py cache --split dev --rttm samples/dev/all_dev.rttm             # caractéristiques + sortie energy_vad
python vad.py cache --split test --rttm samples/test/sample90_seed130.rttm
python vad.py eval --method reference                                        # 6.649 %
python vad.py eval --method best                                             # 3.131 % (dev-train)
python vad.py eval --method reference --split test --rttm samples/test/sample90_seed130.rttm
python vad.py eval --method best --split test --rttm samples/test/sample90_seed130.rttm                   # 5.761 %
python vad.py eval --method best --split test --rttm samples/test/sample90_seed130.rttm --set train=alldev # 5.434 %
python vad.py                                                                # main() d'origine (balayage delta, écrit figures/)
```
Utilisation dans le pipeline :
```
from vad import best_vad, load_audio
timeline = best_vad(load_audio(chemin_wav))
```
Les modèles sont relus depuis `results/vad_agent/models/` ; s'ils sont absents, ils sont réentraînés automatiquement (le cache dev
est alors nécessaire).

Contenu de `results/vad_agent/` (779 Mo) :
- `cache/dev` : 405 Mo ;
- `cache/test` : 355 Mo ;
- `models/` : 6 fichiers, environ 21 Mo ;
- `journal.md`, `rapport.md` et 5 fichiers CSV.
