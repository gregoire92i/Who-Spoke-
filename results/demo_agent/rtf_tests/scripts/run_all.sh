set -e
cd /home/grego/projets/Who-Spoke-; source .venv/bin/activate
B="python /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/bench.py"
# 1. formats
for p in baseline pyannote; do for c in afjiv_13s afjiv_full; do for fm in wav flac mp3 m4a; do
  $B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/formats.csv $c $p 4 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/$c.$fm; done; done; done
# pad test
for p in baseline pyannote; do $B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/formats.csv afjiv_13s_pad $p 4 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/afjiv_13s_pad13056.wav; done
# 2. repetitions
for p in baseline pyannote; do for fm in wav m4a; do $B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/repeat.csv afjiv_13s $p 10 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/afjiv_13s.$fm; done; done
# 3. durees
for p in baseline pyannote; do for d in 5 10 13 20 30 60 120 300; do $B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/durations.csv ipqqq_${d}s $p 6 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/ipqqq_${d}s.wav; done; done
$B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/durations.csv ldnro baseline 6 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/ldnro.wav
$B /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/durations.csv ldnro pyannote 3 /tmp/claude-1000/-home-grego-projets-Who-Spoke-/cce76fd4-990c-424b-b2c1-c3193147073f/scratchpad/audio/ldnro.wav
echo ALLDONE
