import experimenations as signal
from rttm_files import load_audios_from_annotation
import metrics_and_sanity_checks as checks


#parameters
sr=16000#frequence d'echantillonage
frame_size=64#en ms
hop_size=10#en ms
#conversion en nombre/s
frame_size=int(frame_size*(sr/1000))
hop_size=int(hop_size*(sr/1000))
import experimenations as signal
from rttm_files import load_audios_from_annotation, load_file_annotations
import metrics_and_sanity_checks as checks
from pyannote.database.util import load_rttm

#O load annotations from the sample
sample_annotations=load_rttm("samples/sample90_seed130.rttm")

#1 echantillonage
audios=load_audios_from_annotation(sample_annotations,sr,"data/voxconverse/dev/audio")

#2 VAD
