"""Tests de la démo (demo/). Lancement depuis la racine du dépôt :  python -m pytest tests/test_demo.py -v
Fichiers audio utilisés : uniquement le DEV de VoxConverse (jamais le test). Les tests qui en ont besoin sont
sautés si data/voxconverse/dev/ est absent. Les fichiers générés vont dans le dossier temporaire de pytest."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # vad.py lit ses modèles avec des chemins relatifs à la racine

import demo  # noqa: E402  (fixe HF_HUB_OFFLINE, MPLBACKEND… avant les autres imports)
from demo import core  # noqa: E402
from pyannote.core import Annotation, Segment, Timeline  # noqa: E402
from pyannote.database.util import load_rttm  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

DEV_AUDIO = ROOT / "data" / "voxconverse" / "dev" / "audio"
DEV_ANN = ROOT / "data" / "voxconverse" / "annotations" / "dev"
EQUALITY_FILES = ["hqyok", "tucrg"]  # 2 fichiers dev courts (22,0 s et 26,2 s)
needs_dev = pytest.mark.skipif(not DEV_AUDIO.exists(), reason="audio dev VoxConverse absent")


def synthetic_annotation(uri="synth"):
    """3 locuteurs, avec overlap : A [0, 5] et [10, 12], B [3, 8], C [11, 11.5]."""
    ann = Annotation(uri=uri)
    ann[Segment(0, 5)] = "A"
    ann[Segment(3, 8)] = "B"
    ann[Segment(10, 12)] = "A"
    ann[Segment(11, 11.5)] = "C"
    return ann


def der_pair(ref, hyp, duration):
    uem = Timeline([Segment(0, duration)], uri=ref.uri)
    return (DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem),
            DiarizationErrorRate(collar=0.0, skip_overlap=True)(ref, hyp, uem=uem))


# ------------------------------------------------------------------ RTTM
def test_rttm_roundtrip_synthetic(tmp_path):
    ann = synthetic_annotation()
    path = core.to_rttm(ann, "synth", tmp_path / "synth.rttm")
    back = load_rttm(str(path))["synth"]
    assert der_pair(ann, back, 12.0) == (0.0, 0.0)
    for line in path.read_text().splitlines():  # format : 10 champs séparés par des espaces
        f = line.split()
        assert len(f) == 10 and f[0] == "SPEAKER" and f[1] == "synth" and f[5] == "<NA>"


@needs_dev
def test_rttm_roundtrip_pipeline_output(tmp_path):
    """Même idée que le sanity check 1 (D-37), sur une vraie sortie de la baseline."""
    from demo import baseline_adapter
    audio = core.load_audio(DEV_AUDIO / "afjiv.wav")
    hyp, _, _ = baseline_adapter.diarize(audio["pipeline_path"], audio["uri"])
    assert len(hyp) > 0
    back = load_rttm(str(core.to_rttm(hyp, audio["uri"], tmp_path / "afjiv.rttm")))[audio["uri"]]
    # le RTTM arrondit à la ms (write_rttm : 3 décimales) ; les frontières de la baseline sont des multiples du
    # hop de 32,5 ms (4 décimales). Début et durée sont arrondis séparément : écart <= 0,5 ms sur le début,
    # <= 1 ms sur la fin, d'où un DER non nul mais ~1e-4
    t1, t2 = list(hyp.itertracks(yield_label=True)), list(back.itertracks(yield_label=True))
    assert len(t1) == len(t2)
    for (s1, _, l1), (s2, _, l2) in zip(t1, t2):
        assert l1 == l2 and abs(s1.start - s2.start) <= 5e-4 + 1e-9 and abs(s1.end - s2.end) <= 1e-3 + 1e-9
    der, der_f = der_pair(hyp, back, audio["duration"])
    assert der < 1e-3 and der_f < 1e-3
    # une fois arrondie, l'annotation fait un aller-retour exact (DER = 0, comme le sanity check 1)
    back2 = load_rttm(str(core.to_rttm(back, audio["uri"], tmp_path / "afjiv2.rttm")))[audio["uri"]]
    assert der_pair(back, back2, audio["duration"]) == (0.0, 0.0)


def test_rttm_empty_annotation(tmp_path):
    path = core.to_rttm(Annotation(uri="vide"), "vide", tmp_path / "vide.rttm")
    assert path.exists() and path.read_text() == ""
    assert load_rttm(str(path)) == {}


def test_make_uri():
    assert core.make_uri("/a/b/mon enregistrement (1).m4a") == "mon_enregistrement__1_"
    assert core.make_uri("abjxc.wav") == "abjxc"
    assert core.make_uri("réunion inconnue.m4a") == "reunion_inconnue"
    assert core.make_uri("日本.wav") == "audio"  # rien de latin : nom par défaut


# ------------------------------------------------------------------ temps de parole
def test_speaker_stats_consistent():
    ann = synthetic_annotation()
    rows, summary = core.speaker_stats(ann, duration=20.0)
    by = {r["label"]: r for r in rows}
    assert by["A"]["speech_s"] == pytest.approx(7.0) and by["A"]["n_segments"] == 2
    assert by["B"]["speech_s"] == pytest.approx(5.0) and by["C"]["speech_s"] == pytest.approx(0.5)
    assert summary["speech_s"] == pytest.approx(10.0)  # [0, 8] + [10, 12]
    assert summary["overlap_s"] == pytest.approx(2.5)  # [3, 5] + [11, 11.5]
    # somme des temps par locuteur = parole + overlap (au plus 2 locuteurs simultanés ici)
    assert sum(r["speech_s"] for r in rows) == pytest.approx(summary["speech_s"] + summary["overlap_s"])
    assert sum(r["speech_s"] for r in rows) == pytest.approx(sum(ann.label_duration(l) for l in ann.labels()))
    assert [r["label"] for r in rows] == ["A", "B", "C"]  # tri par temps décroissant
    assert by["A"]["pct_audio"] == pytest.approx(35.0)


def test_speaker_stats_empty():
    rows, summary = core.speaker_stats(Annotation(uri="x"), duration=5.0)
    assert rows == [] and summary["speech_s"] == 0 and summary["n_speakers"] == 0


# ------------------------------------------------------------------ extraction de segments
def test_segment_audio_bounds():
    sr = 16000
    y = np.arange(10 * sr, dtype=np.float32)  # 10 s ; y[i] = i
    seg = core.segment_audio(y, sr, Segment(1.0, 2.5))
    assert len(seg) == int(1.5 * sr) and seg[0] == sr and seg[-1] == 2.5 * sr - 1
    seg = core.segment_audio(y, sr, (9.5, 12.0))  # dépasse la fin du fichier
    assert len(seg) == sr // 2 and seg[-1] == len(y) - 1
    seg = core.segment_audio(y, sr, (0.2, 1.0), margin=0.5)  # marge coupée au début
    assert seg[0] == 0 and len(seg) == int(1.5 * sr)
    assert len(core.segment_audio(y, sr, (11.0, 12.0))) == 0  # entièrement après la fin


# ------------------------------------------------------------------ chargement audio
@needs_dev
def test_load_audio_formats(tmp_path):
    src = DEV_AUDIO / "hqyok.wav"
    y, sr = sf.read(src)
    ref = core.load_audio(src)
    assert ref["sr"] == 16000 and not ref["converted"] and ref["duration"] == pytest.approx(sf.info(src).duration)
    stereo = tmp_path / "stereo 44k.flac"  # stéréo, 44,1 kHz, nom avec espace
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "2", "-ar", "44100", str(stereo)],
                   check=True)
    a = core.load_audio(stereo)
    assert a["original_channels"] == 2 and a["original_sr"] == 44100 and a["uri"] == "stereo_44k"
    assert a["y"].ndim == 1 and a["duration"] == pytest.approx(ref["duration"], abs=0.01)
    m4a = tmp_path / "rec.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), str(m4a)], check=True)
    b = core.load_audio(m4a)
    assert b["converted"] and b["pipeline_path"].suffix == ".wav"
    assert demo.TMP_DIR in b["pipeline_path"].parents  # conversion dans demo/tmp/ (ignoré par git)
    assert b["duration"] == pytest.approx(ref["duration"], abs=0.1)
    b["pipeline_path"].unlink()
    mp3 = tmp_path / "rec48k.mp3"  # mp3 : converti en wav (pyannote échoue sur la lecture directe du mp3)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", "48000", str(mp3)], check=True)
    c = core.load_audio(mp3)
    assert c["converted"] and c["original_sr"] == 48000 and c["pipeline_path"].suffix == ".wav"
    assert sf.info(c["pipeline_path"]).samplerate == 16000 and sf.info(c["pipeline_path"]).channels == 1
    c["pipeline_path"].unlink()


def test_load_audio_errors(tmp_path):
    bad = tmp_path / "corrompu.wav"
    bad.write_bytes(b"RIFF" + b"\x00" * 2000)
    txt = tmp_path / "texte.mp3"
    txt.write_text("ceci n'est pas un audio")
    empty = tmp_path / "vide.wav"
    empty.write_bytes(b"")
    for p in (bad, txt, empty, tmp_path / "absent.wav"):
        with pytest.raises(core.AudioLoadError):
            core.load_audio(p)


# ------------------------------------------------------------------ cohérence avec baseline.py
@needs_dev
@pytest.mark.parametrize("pretrained", [False, True], ids=["mfcc", "wespeaker"])
@pytest.mark.parametrize("uri", EQUALITY_FILES)
def test_adapter_equals_baseline_devtrain(uri, pretrained):
    """Avec le même vad_train et le même choix d'embeddings, la sortie de l'adaptateur est identique à celle de
    baseline.diarize_file (les deux modes : MFCC et WeSpeaker)."""
    import baseline
    from demo import baseline_adapter
    path = DEV_AUDIO / f"{uri}.wav"
    ref_hyp, _, _ = baseline.diarize_file(path, uri, frame_size=baseline.frame_size, hop_size=baseline.hop_size,
                                          vad_train="devtrain", pretrained_embeddings=pretrained)
    hyp, params, extra = baseline_adapter.diarize(path, uri, vad_train="devtrain", pretrained_embeddings=pretrained)
    assert list(hyp.itertracks(yield_label=True)) == list(ref_hyp.itertracks(yield_label=True))
    assert params["VAD"] == "best_vad(train='devtrain')"
    assert extra["pretrained_embeddings"] is pretrained
    assert params["embeddings"].startswith("WeSpeaker" if pretrained else "statistiques de MFCC")


def test_embeddings_default_is_mfcc():
    """Défaut de la démo (décision de Grégoire du 16/11) : statistiques de MFCC, y compris par core.diarize."""
    import inspect
    from demo import baseline_adapter
    assert baseline_adapter.DEFAULT_PRETRAINED is False
    assert inspect.signature(baseline_adapter.diarize).parameters["pretrained_embeddings"].default is False
    assert inspect.signature(core.diarize).parameters["pretrained_embeddings"].default is False
    assert baseline_adapter.params()["embeddings"] == "statistiques de MFCC"


@needs_dev
def test_option_reaches_wespeaker(monkeypatch):
    """Garde-fou : l'option est bien transmise. On espionne baseline.compute_pretrained_embeddings (sans changer son
    résultat) : appelée 0 fois en mode MFCC, 1 fois en mode WeSpeaker, sur les mêmes windows."""
    import baseline
    from demo import baseline_adapter
    calls = []
    real = baseline.compute_pretrained_embeddings

    def spy(audio, win_segs, **kw):
        X = real(audio, win_segs, **kw)
        calls.append(X.shape)
        return X
    monkeypatch.setattr(baseline, "compute_pretrained_embeddings", spy)
    path = DEV_AUDIO / f"{EQUALITY_FILES[0]}.wav"
    _, _, ex_m = baseline_adapter.diarize(path, EQUALITY_FILES[0], pretrained_embeddings=False)
    assert calls == []
    _, _, ex_w = baseline_adapter.diarize(path, EQUALITY_FILES[0], pretrained_embeddings=True)
    assert calls == [(ex_w["n_windows"], 256)]
    assert ex_m["n_windows"] == ex_w["n_windows"]  # mêmes windows dans les deux modes


def test_wespeaker_preloaded_offline():
    """WeSpeaker est préchargé au démarrage, sans réseau (HF_HUB_OFFLINE=1 fixé par demo/__init__.py)."""
    import embeddings
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    engines = core.Engines(load_pyannote=False, log=lambda m: None)
    assert engines.wespeaker[0], engines.wespeaker[1]
    assert any(k[0] == embeddings.WESPEAKER_ID for k in embeddings._pretrained_cache)


def test_wespeaker_unavailable_refused():
    """Si WeSpeaker n'a pas pu être chargé, l'analyse WeSpeaker est refusée avec un message clair (pas de
    téléchargement ni de bascule silencieuse vers les MFCC)."""
    engines = core.Engines(load_pyannote=False, load_baseline=False, log=lambda m: None)
    engines.status["baseline"] = (True, "simulé")
    engines.wespeaker = (False, "WeSpeaker indisponible : simulé")
    with pytest.raises(RuntimeError, match="WeSpeaker indisponible"):
        core.diarize({"pipeline_path": None, "uri": "x", "duration": 1.0}, "baseline", engines,
                     pretrained_embeddings=True)


@needs_dev
def test_demo_uses_alldev():
    """La démo appelle baseline.diarize_file avec vad_train='alldev' (modèles de production)."""
    import baseline
    from demo import baseline_adapter
    assert baseline_adapter.VAD_TRAIN == "alldev"
    path = DEV_AUDIO / "hqyok.wav"
    ref_hyp, _, _ = baseline.diarize_file(path, "hqyok", frame_size=baseline.frame_size, hop_size=baseline.hop_size,
                                          vad_train="alldev")
    audio = core.load_audio(path)
    engines = core.Engines(load_pyannote=False)
    hyp, infos = core.diarize(audio, "baseline", engines)
    assert list(hyp.itertracks(yield_label=True)) == list(ref_hyp.itertracks(yield_label=True))
    assert infos["params"]["VAD"] == "best_vad(train='alldev')"
    assert infos["RTF"] == pytest.approx(infos["processing_time_s"] / infos["duration_s"])


def test_vad_models_present():
    from demo import baseline_adapter
    assert baseline_adapter.check_vad_models()[0] == "ok"


# ------------------------------------------------------------------ référence, renommage
def test_evaluate_reference_against_itself():
    ann = synthetic_annotation()
    m = core.evaluate(ann, ann.copy(), duration=15.0)
    assert m["DER"] == 0 and m["DER_forgiving"] == 0 and m["JER"] == 0 and m["confusion"] == 0


def test_rename():
    ann = synthetic_annotation()
    out = core.rename(ann, {"A": "Alice Martin", "B": ""})
    assert sorted(out.labels()) == ["Alice_Martin", "B", "C"]
    assert sorted(ann.labels()) == ["A", "B", "C"]  # l'original n'est pas modifié
    with pytest.raises(ValueError):
        core.rename(ann, {"A": "X", "B": "X"})


def test_vad_models_missing_states(tmp_path, monkeypatch):
    """Modèles absents (simulé : dossier de modèles vide) : 'retrain' si les prérequis sont là (ce PC),
    'missing' sinon (racine simulée vide), avec la commande de régénération ; baseline alors indisponible."""
    from demo import baseline_adapter
    monkeypatch.setattr(baseline_adapter, "MODELS_DIR", tmp_path / "models")
    if (ROOT / "results/vad_agent/cache/dev").exists():
        state, msg = baseline_adapter.check_vad_models()
        assert state == "retrain" and "réentraînement automatique" in msg
    monkeypatch.setattr(baseline_adapter, "ROOT", tmp_path)
    state, msg = baseline_adapter.check_vad_models()
    assert state == "missing" and baseline_adapter.RETRAIN_CMD in msg
    engines = core.Engines(load_pyannote=False, log=lambda m: None)
    assert engines.available() == [] and engines.status["baseline"][0] is False
    with pytest.raises(RuntimeError):
        core.diarize({"pipeline_path": None, "uri": "x", "duration": 1.0}, "baseline", engines)
