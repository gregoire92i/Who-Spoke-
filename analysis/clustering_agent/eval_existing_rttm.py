"""Réévalue les RTTM existants (sans les modifier) : DER moderne / forgiving / JER micro + FA/miss/confusion.
Usage : python analysis/clustering_agent/eval_existing_rttm.py fichier1.rttm [fichier2.rttm ...]
(un argument 'HEAD:chemin' lit la version committée via git show)."""
import sys, subprocess, tempfile
from pathlib import Path
import soundfile as sf
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

REF = load_rttm("samples/dev/sample90_seed130.rttm")
AUDIO = Path("data/voxconverse/dev/audio")

def evaluate(hyps):
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    derf = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    for uri in sorted(REF):
        dur = sf.info(str(AUDIO / f"{uri}.wav")).duration
        uem = Timeline([Segment(0, dur)], uri=uri)
        from pyannote.core import Annotation
        hyp = hyps.get(uri, Annotation(uri=uri))
        der(REF[uri], hyp, uem=uem); derf(REF[uri], hyp, uem=uem); jer(REF[uri], hyp, uem=uem)
    a = der.accumulated_
    t = a["total"]
    return dict(DER=abs(der), DER_forgiving=abs(derf), JER=abs(jer), FA=a["false alarm"] / t,
                missed=a["missed detection"] / t, confusion=a["confusion"] / t)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg.startswith("HEAD:"):
            txt = subprocess.run(["git", "show", arg], capture_output=True, text=True, check=True).stdout
            tmp = tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False); tmp.write(txt); tmp.close()
            hyps = load_rttm(tmp.name)
        else:
            hyps = load_rttm(arg)
        r = evaluate(hyps)
        print(f"{arg}\n  DER {r['DER']:.2%} | forgiving {r['DER_forgiving']:.2%} | JER {r['JER']:.2%} | "
              f"FA {r['FA']:.2%} | missed {r['missed']:.2%} | confusion {r['confusion']:.2%}")
