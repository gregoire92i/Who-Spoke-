"""Cœur de la démo, indépendant de l'interface.

- load_audio : n'importe quel format -> signal 16 kHz mono (D-50) + chemin lisible par les pipelines ;
- diarize : point d'entrée unique pour les deux pipelines (baseline, pyannote) -> (Annotation, infos) ;
- speaker_stats, segments_table, segment_audio : temps de parole, tableau des segments, extraction pour l'écoute ;
- to_rttm : export RTTM (Annotation.write_rttm de pyannote.core) ;
- evaluate : DER / DER forgiving / JER si un RTTM de référence est fourni (protocole D-30 à D-37).

Aucun paramètre de pipeline n'est défini ici : la baseline lit ses valeurs dans baseline.py, pyannote dans
pyannote_diarization.py. Un seul fichier est traité à la fois."""
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline

from demo import ROOT, TMP_DIR

SR = 16000  # D-50
PIPELINES = ("baseline", "pyannote")
# formats proposés dans l'interface ; tout format décodable par ffmpeg est en fait accepté
AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"]
# conteneurs sans perte passés tels quels aux pipelines (formats soundfile) ; tout le reste (mp3, ogg, m4a…) est
# d'abord converti en wav 16 kHz mono par ffmpeg (D-50). Raison : pyannote lit le mp3 par morceaux et échoue
# (« requested chunk … resulted in 478895 samples instead of the expected 480000 », observé sur un mp3 48 kHz).
DIRECT_FORMATS = {"WAV", "WAVEX", "FLAC"}


class AudioLoadError(Exception):
    """Fichier absent, vide, corrompu ou dans un format que ni soundfile ni ffmpeg ne savent lire."""


# ------------------------------------------------------------------ audio
def make_uri(path):
    """URI = nom du fichier sans extension, accents retirés ; tout caractère hors [A-Za-z0-9_-] devient '_' (un espace
    casserait le RTTM, dont les champs sont séparés par des espaces)."""
    stem = unicodedata.normalize("NFKD", Path(path).stem).encode("ascii", "ignore").decode()  # é -> e
    uri = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return uri or "audio"


def _ffmpeg_to_wav(path, uri):
    """Conversion en wav 16 kHz mono PCM 16 bits dans demo/tmp/ (D-50). Renvoie le chemin du wav."""
    if shutil.which("ffmpeg") is None:
        raise AudioLoadError("ffmpeg est introuvable : impossible de lire ce format (installer le paquet ffmpeg).")
    out_dir = TMP_DIR / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uri}_{uuid.uuid4().hex[:8]}.wav"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
           "-c:a", "pcm_s16le", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        out.unlink(missing_ok=True)
        msg = (res.stderr or "").strip().splitlines()
        raise AudioLoadError(f"Format non supporté ou fichier corrompu ({Path(path).name}) : "
                             f"{msg[-1] if msg else 'ffmpeg n a produit aucun audio'}")
    return out


def load_audio(path):
    """Charge un fichier audio.

    Renvoie un dict :
      y : signal float32 16 kHz mono (librosa, D-50) ;
      sr : 16000 ;
      duration : durée réelle du signal (s) = len(y) / sr ;
      pipeline_path : chemin passé aux pipelines. C'est le fichier d'origine si c'est un wav ou un flac (la baseline
        le traite alors exactement comme baseline.py). Sinon (mp3, ogg, m4a, aac…), c'est un wav 16 kHz mono
        converti par ffmpeg dans demo/tmp/converted/ (voir DIRECT_FORMATS) ;
      uri, converted (bool), original_sr, original_channels (None si le fichier a été converti)."""
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Fichier introuvable : {path.name}")
    if path.stat().st_size == 0:
        raise AudioLoadError(f"Fichier vide : {path.name}")
    uri = make_uri(path)
    converted, original_sr, original_channels = False, None, None
    try:
        info = sf.info(str(path))
        original_sr, original_channels = info.samplerate, info.channels
        lossless = info.format in DIRECT_FORMATS
    except Exception:  # format non lu par libsndfile (m4a, aac…) ou fichier corrompu : on tente ffmpeg
        lossless = False
    if lossless:
        pipeline_path = path
    else:
        pipeline_path = _ffmpeg_to_wav(path, uri)
        converted = True
    try:
        y, _ = librosa.load(str(pipeline_path), sr=SR, mono=True)
    except Exception as e:
        raise AudioLoadError(f"Impossible de décoder {path.name} : {type(e).__name__}") from e
    if y.size == 0:
        raise AudioLoadError(f"Le fichier {path.name} ne contient aucun échantillon audio.")
    return {"y": y, "sr": SR, "duration": len(y) / SR, "pipeline_path": Path(pipeline_path), "uri": uri,
            "converted": converted, "original_sr": original_sr, "original_channels": original_channels}


# ------------------------------------------------------------------ modèles
class Engines:
    """Modèles chargés UNE fois au démarrage et réutilisés pour chaque fichier.
    status[pipeline] = (disponible: bool, message)."""

    def __init__(self, load_pyannote=True, load_baseline=True, log=print):
        self.status = {}
        self.pyannote = None
        self.device = None
        if load_baseline:
            from demo import baseline_adapter
            self.status["baseline"] = baseline_adapter.prepare(log=log)
        if load_pyannote:
            from demo import pyannote_adapter
            try:
                t0 = time.perf_counter()
                self.pyannote, self.device = pyannote_adapter.load(log=log)
                self.status["pyannote"] = (True, f"pyannote chargé sur {self.device} "
                                                 f"en {time.perf_counter() - t0:.1f} s")
            except Exception as e:  # modèle absent du cache, pas de GPU utilisable… : la baseline reste disponible
                self.status["pyannote"] = (False, f"pyannote indisponible : {type(e).__name__}: {e}")
        for p, (ok, msg) in self.status.items():
            log(f"[{p}] {msg}")

    def available(self):
        return [p for p in PIPELINES if self.status.get(p, (False,))[0]]


def diarize(audio, pipeline, engines):
    """Point d'entrée unique. audio : sortie de load_audio. Renvoie (Annotation, infos).
    infos : pipeline, paramètres effectivement utilisés, temps de traitement, durée audio, RTF (D-36).
    Le temps de traitement inclut la lecture du fichier par la pipeline (comme baseline.py), pas load_audio."""
    if pipeline not in engines.available():
        raise RuntimeError(engines.status.get(pipeline, (False, f"pipeline inconnue : {pipeline}"))[1])
    t0 = time.perf_counter()
    if pipeline == "baseline":
        from demo import baseline_adapter
        ann, params, extra = baseline_adapter.diarize(audio["pipeline_path"], audio["uri"])
    else:
        from demo import pyannote_adapter
        ann, params, extra = pyannote_adapter.diarize(engines.pyannote, audio["pipeline_path"])
        params["device"] = str(engines.device)
    proc = time.perf_counter() - t0
    ann.uri = audio["uri"]  # pyannote prend le nom brut du fichier : on impose l'URI nettoyée
    infos = {"pipeline": pipeline, "params": params, **extra, "processing_time_s": proc,
             "duration_s": audio["duration"], "RTF": proc / audio["duration"], "k_hyp": len(ann.labels())}
    return ann, infos


# ------------------------------------------------------------------ analyses
def speaker_stats(annotation, duration):
    """Temps de parole par locuteur. Renvoie (lignes, résumé).
    lignes : une par locuteur, triées par temps décroissant : label, speech_s, pct_speech (part de la durée de
      parole totale), pct_audio (part de la durée audio), n_segments.
    résumé : speech_s (parole, overlap compté une fois), overlap_s, duration_s, n_speakers.
    Avec de l'overlap, la somme des speech_s dépasse la durée de parole (chaque locuteur compte son temps)."""
    speech = annotation.get_timeline().support().duration()
    overlap = annotation.get_overlap().duration()
    rows = []
    for label in annotation.labels():
        t = annotation.label_duration(label)
        rows.append({"label": label, "speech_s": t,
                     "pct_speech": 100 * t / speech if speech > 0 else 0.0,
                     "pct_audio": 100 * t / duration if duration > 0 else 0.0,
                     "n_segments": len(list(annotation.label_timeline(label, copy=False)))})
    rows.sort(key=lambda r: -r["speech_s"])
    return rows, {"speech_s": speech, "overlap_s": overlap, "duration_s": duration, "n_speakers": len(rows)}


def segments_table(annotation):
    """Segments triés par début : (start, end, duration, label)."""
    return [(s.start, s.end, s.duration, label) for s, _, label in annotation.itertracks(yield_label=True)]


def segment_audio(y, sr, segment, margin=0.0):
    """Extrait [start - margin, end + margin] du signal, bornes coupées à [0, len(y)].
    segment : pyannote Segment ou couple (start, end) en secondes."""
    start, end = (segment.start, segment.end) if isinstance(segment, Segment) else segment
    a = max(0, int(round((start - margin) * sr)))
    b = min(len(y), int(round((end + margin) * sr)))
    return y[a:b] if b > a else y[:0]


def rename(annotation, names):
    """Renommage manuel des locuteurs. names : {ancien label: nouveau nom}. Les noms vides sont ignorés ;
    les espaces deviennent '_' (le label est un champ du RTTM). Deux locuteurs ne peuvent pas recevoir le même nom."""
    mapping = {}
    for old, new in names.items():
        new = re.sub(r"\s+", "_", str(new or "").strip())
        if new and new != old:
            mapping[old] = new
    final = [mapping.get(lab, lab) for lab in annotation.labels()]
    if len(set(final)) != len(final):
        raise ValueError("Deux locuteurs auraient le même nom : renommage refusé.")
    return annotation.rename_labels(mapping=mapping) if mapping else annotation.copy()


def to_rttm(annotation, uri, path):
    """Écrit l'annotation au format RTTM (Annotation.write_rttm, pyannote.core 6.1.0 : write_rttm(file)).
    Une annotation vide donne un fichier vide (RTTM valide)."""
    ann = annotation.copy()
    ann.uri = uri
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        ann.write_rttm(f)
    return path


# ------------------------------------------------------------------ référence (option a)
def load_reference(path, uri):
    """Lit un RTTM de référence (load_rttm, D-13). Si le RTTM contient plusieurs URI, prend celle du fichier."""
    from pyannote.database.util import load_rttm
    refs = load_rttm(str(path))
    if not refs:
        raise ValueError("RTTM de référence vide ou illisible.")
    if uri in refs:
        ref = refs[uri]
    elif len(refs) == 1:
        ref = next(iter(refs.values()))
    else:
        raise ValueError(f"Le RTTM contient {len(refs)} fichiers et aucun ne s'appelle {uri}.")
    ref.uri = uri
    return ref


def evaluate(reference, hypothesis, duration):
    """Protocole du projet (D-30 à D-37) : collar 0, UEM = [0, durée réelle de l'audio], DER moderne
    (skip_overlap=False), DER forgiving (skip_overlap=True), JER, décomposition FA / missed / confusion."""
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    hyp = hypothesis.copy()
    hyp.uri = reference.uri
    uem = Timeline([Segment(0, duration)], uri=reference.uri)
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    d = der(reference, hyp, uem=uem, detailed=True)
    total = d["total"]
    out = {"DER": d["diarization error rate"],
           "DER_forgiving": DiarizationErrorRate(collar=0.0, skip_overlap=True)(reference, hyp, uem=uem),
           "JER": JaccardErrorRate(collar=0.0, skip_overlap=False)(reference, hyp, uem=uem),
           "false_alarm": d["false alarm"] / total if total else 0.0,
           "missed": d["missed detection"] / total if total else 0.0,
           "confusion": d["confusion"] / total if total else 0.0,
           "ref_speech_s": total, "n_ref_speakers": len(reference.labels())}
    return out


def fmt_time(t):
    """Secondes -> m:ss.s (affichage)."""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"
