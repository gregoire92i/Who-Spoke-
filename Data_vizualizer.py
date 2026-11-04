"""Per-file statistics of the VoxConverse corpus (dev and test), computed from RTTM files.
 
Run from the repository root:
    python plot_corpus_stats.py
 
Outputs (in figures/):
    voxconverse_file_stats.csv   one row per file
    voxconverse_distributions.png one panel per metric, every dot is a file
    voxconverse_pair_grid.png     relationships between metrics, every dot is a file
"""

###THIS FILE WAS CODED with CLAUDE


from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # no display available: figures are only saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
 
DATA_DIR = Path("data/voxconverse")
ANNOTATION_DIR = DATA_DIR / "annotations"
FIGURE_DIR = Path("figures")
SPLITS = ["dev", "test"]
SPLIT_COLORS = {"dev": "tab:blue", "test": "tab:orange"}
 
# column name -> axis label (change this dict to plot other columns, e.g. "file_size_mb")
METRICS = {
    "overlap_ratio_pct": "Overlap (% of speech time)",
    "n_speakers": "Number of speakers",
    "speech_ratio_pct": "Speech (% of file duration)",
    "audio_duration_min": "File duration (min)",
}
RANDOM_SEED = 0  # only used for the horizontal jitter of the dots
 
 
def find_wav_files(split_dir):
    """Return {file_id: wav_path}, ignoring macOS metadata files (._*.wav, __MACOSX/)."""
    wav_paths = {}
    for wav_path in split_dir.rglob("*.wav"):
        if wav_path.name.startswith("._") or "__MACOSX" in wav_path.parts:
            continue
        wav_paths[wav_path.stem] = wav_path
    return wav_paths
 
 
def load_reference(rttm_path):
    """Return the Annotation of one VoxConverse RTTM file (one uri per file)."""
    annotations = load_rttm(str(rttm_path))
    if rttm_path.stem in annotations:
        return annotations[rttm_path.stem]
    if annotations:
        return next(iter(annotations.values()))
    return Annotation(uri=rttm_path.stem)  # empty RTTM file
 
 
def compute_file_stats(rttm_path, wav_path):
    """Compute the statistics of one file from its RTTM and its wav header."""
    annotation = load_reference(rttm_path)
    audio_info = sf.info(str(wav_path))  # reads the header only, not the samples
    audio_duration = audio_info.duration  # seconds
 
    # support() merges overlapping segments: overlap is counted only once
    speech_duration = annotation.get_timeline().support().duration()
    overlap_duration = annotation.get_overlap().duration()
 
    return {
        "file_id": rttm_path.stem,
        "n_speakers": len(annotation.labels()),
        "audio_duration_min": audio_duration / 60,
        "file_size_mb": wav_path.stat().st_size / 1e6,
        "sample_rate": audio_info.samplerate,
        "n_channels": audio_info.channels,
        # same definition as the VoxConverse authors: overlap / speech time
        "overlap_ratio_pct": 100 * overlap_duration / speech_duration if speech_duration > 0 else 0.0,
        "speech_ratio_pct": 100 * speech_duration / audio_duration,
    }
 
 
def build_stats_table():
    """Return a DataFrame with one row per file, for all splits."""
    rows = []
    for split in SPLITS:
        wav_paths = find_wav_files(DATA_DIR / split)
        rttm_paths = sorted((ANNOTATION_DIR / split).glob("*.rttm"))
        missing_wav = []
        for rttm_path in rttm_paths:
            wav_path = wav_paths.get(rttm_path.stem)
            if wav_path is None:
                missing_wav.append(rttm_path.stem)
                continue
            file_stats = compute_file_stats(rttm_path, wav_path)
            file_stats["split"] = split
            rows.append(file_stats)
        print(f"[{split}] {len(rttm_paths)} rttm, {len(wav_paths)} wav, "
              f"{len(missing_wav)} rttm without wav {missing_wav[:5]}")
    return pd.DataFrame(rows)
 
 
def plot_distributions(stats_table, output_path):
    """One panel per metric: boxplot per split + one jittered dot per file."""
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))
    for ax, (column, label) in zip(axes, METRICS.items()):
        for position, split in enumerate(SPLITS):
            values = stats_table.loc[stats_table["split"] == split, column].to_numpy()
            ax.boxplot(values, positions=[position], widths=0.5, showfliers=False,
                       medianprops={"color": "black"})
            x_jitter = position + rng.uniform(-0.2, 0.2, size=len(values))
            ax.scatter(x_jitter, values, s=10, alpha=0.5, color=SPLIT_COLORS[split])
            ax.annotate(f"n={len(values)}\nmed={np.median(values):.1f}",
                        xy=(position, 1.01), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(SPLITS)), SPLITS)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("VoxConverse v0.3 - per-file statistics (one dot = one file)", y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_pair_grid(stats_table, output_path):
    """Scatter of every metric against every other one; histograms on the diagonal."""
    columns = list(METRICS)
    n_metrics = len(columns)
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(3 * n_metrics, 3 * n_metrics))
    for row_index, y_column in enumerate(columns):
        for col_index, x_column in enumerate(columns):
            ax = axes[row_index, col_index]
            for split in SPLITS:
                split_table = stats_table[stats_table["split"] == split]
                if row_index == col_index:
                    # density=True so that dev (216 files) and test (232) are comparable
                    ax.hist(split_table[x_column], bins=20, density=True, alpha=0.5,
                            color=SPLIT_COLORS[split], label=split)
                else:
                    ax.scatter(split_table[x_column], split_table[y_column], s=8,
                               alpha=0.5, color=SPLIT_COLORS[split], label=split)
            if row_index == n_metrics - 1:
                ax.set_xlabel(METRICS[x_column])
            if row_index == col_index:
                # diagonal: histogram of the column metric, y axis is a density
                ax.set_title(f"Histogram: {METRICS[x_column]}", fontsize=9, fontweight="bold")
                ax.set_ylabel("density", fontsize=8)
            else:
                # scatter: y axis is always the metric of the row
                ax.set_ylabel(METRICS[y_column], fontsize=8)
    axes[0, 0].legend()
    fig.suptitle("VoxConverse v0.3 - relationships between metrics (one dot = one file)\n"
                 "scatter: y = metric of the row, x = metric of the column")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
 
 
def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    stats_table = build_stats_table()
 
    csv_path = FIGURE_DIR / "voxconverse_file_stats.csv"
    stats_table.to_csv(csv_path, index=False, float_format="%.3f")
 
    # if every wav has the same format, file size in MB is redundant with duration
    print("sample rates:", sorted(stats_table["sample_rate"].unique()),
          "| channels:", sorted(stats_table["n_channels"].unique()))
    print(stats_table.groupby("split")[list(METRICS) + ["file_size_mb"]].describe().T.round(2))
 
    plot_distributions(stats_table, FIGURE_DIR / "voxconverse_distributions.png")
    plot_pair_grid(stats_table, FIGURE_DIR / "voxconverse_pair_grid.png")
    print(f"saved: {csv_path} and 2 figures in {FIGURE_DIR}/")
 
 
if __name__ == "__main__":
    main()




###### SAMPLE vs DEV ######

SAMPLE_RTTM_DIR = "data/voxconverse/annotations/dev"
SAMPLE_WAV_DIR = "data/voxconverse/dev/audio"
SAMPLE_SIZE = 100
sample_seed=42


def plot_sample_vs_dev(sample_ids):
    """Same figures as main(), but comparing the full dev set with the sample (sample files are also in dev)."""
    global SPLITS
    stats_table = pd.read_csv(FIGURE_DIR / "voxconverse_file_stats.csv")  # written by main()
    dev_table = stats_table[stats_table["split"] == "dev"]
    sample_table = dev_table[dev_table["file_id"].isin(sample_ids)].copy()
    sample_table["split"] = "sample"
    print(f"sample: {len(sample_table)} files found in dev (expected {len(sample_ids)})")
    comparison_table = pd.concat([dev_table, sample_table])

    saved_splits = SPLITS
    SPLITS = ["dev", "sample"]  # the plot functions loop over SPLITS
    SPLIT_COLORS["sample"] = "tab:green"
    try:
        plot_distributions(comparison_table, FIGURE_DIR / "sample_vs_dev_distributions.png")
        plot_pair_grid(comparison_table, FIGURE_DIR / "sample_vs_dev_pair_grid.png")
    finally:
        SPLITS = saved_splits
    print(comparison_table.groupby("split")[list(METRICS)].describe().T.round(2))
    print(f"saved: 2 figures sample_vs_dev_*.png in {FIGURE_DIR}/")


if __name__ == "__main__":
    from rttm_files import get_representative_sample
    sample = get_representative_sample(SAMPLE_RTTM_DIR, SAMPLE_WAV_DIR, SAMPLE_SIZE,seed=sample_seed)
    plot_sample_vs_dev(list(sample))