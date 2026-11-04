import soundfile as sf# durée réelle des .wav
from pyannote.core import Segment, Timeline, Annotation
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import (
    DiarizationErrorRate, GreedyDiarizationErrorRate, JaccardErrorRate,
    DiarizationPurity, DiarizationCoverage, DiarizationSpeakerCountError,
)
from pyannote.metrics.detection import (
    DetectionErrorRate, DetectionPrecision, DetectionRecall,
    DetectionAccuracy, DetectionPrecisionRecallFMeasure,
)
from rttm_files import get_representative_sample
from pathlib import Path #pour manipuler les fichiers (supprimer les char du type (.pdf), etc...)

def sanity_check1_ref(annotations_,audio_folder_path, check_collar=0.00):
    """prends en parametres dico d'annotations, renvoie un metrics sanity check1 sur celui-ci : cad compare le tau d'erreur sur lui mme"""
    folder_wav=Path(audio_folder_path)
    metric = DiarizationErrorRate(check_collar,skip_overlap=False)
    annot_ref= annotations_
    annot_ref1=annotations_
    for stem in annot_ref:
        metric(annot_ref[stem],annot_ref1[stem],uem=Timeline([Segment(0,sf.info(folder_wav/f"{stem}.wav").duration)]))#etre sur que l'uem fasse bien toute la durée du wav
    output=metric.report(display=True)
    metric.reset()#vider le tableau de metriques
    return output

def sanity_check2_1speaker(annotations_,audio_folder_path, check_collar=0.00):
    """prends en parametres dico d'annotations, renvoie un metrics sanity check2 sur celui-ci : cad compare les annotations à un seul speaker sur chaque fichier sur 100% de du file size"""
    folder_wav=Path(audio_folder_path)
    metric = DiarizationErrorRate(check_collar,skip_overlap=False)
    for stem in annotations_:
        records=[(Segment(0,sf.info(folder_wav/f"{stem}.wav").duration),0,"spk00")]
        annot_hyp=Annotation.from_records(records, uri=stem, modality=None)
        metric(annotations_[stem],annot_hyp,uem=Timeline([Segment(0,sf.info(folder_wav/f"{stem}.wav").duration)]))
    output=metric.report(display=True)
    metric.reset()#vider le tableau de metriques
    return output

##TESTS##
# sanity_check1_ref(get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", 40), "data/voxconverse/dev/audio")
# sanity_check2_1speaker(get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", 40), "data/voxconverse/dev/audio")