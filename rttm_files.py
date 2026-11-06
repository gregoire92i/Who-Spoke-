from pyannote.core import notebook  #pr structure des données (segment, annotations, timeline)
from pyannote.database.util import load_rttm #chargement des fichiers 
from pathlib import Path #pour manipuler les fichiers (supprimer les char du type (.pdf), etc...)
import numpy as np
import matplotlib.pyplot as plt#affichage
import pandas as pd
from scipy.stats import ks_2samp
import soundfile as sf
import random
import soundfile
from pyannote.database.util import load_rttm

def load_file_annotations(rttm_file):
    rttm_file=Path(rttm_file)
    annotation=load_rttm(rttm_file)[rttm_file.stem]
    return annotation

def load_audios_from_annotation(annotation_, sr, audio_dir, ext=".wav"):
    "takes an annotation of audios in parameters, return a dico_audio[stem]=audio_content"
    from experimenations import load_audio

    # Un Annotation seul -> on le met dans un dict pour traiter les deux cas pareil
    if not isinstance(annotation_, dict):
        annotation_ = {annotation_.uri: annotation_}

    audio_dir = Path(audio_dir)
    dico_audio = {}
    missing = []

    for stem in annotation_:              # uri RTTM = nom du fichier sans extension
        path = audio_dir / f"{stem}{ext}"
        if not path.exists():
            missing.append(stem)
            continue
        dico_audio[stem] = load_audio(str(path), sr)

    if missing:
        print(f"[warning] {len(missing)} audio(s) introuvable(s) : {missing}")

    return dico_audio


#graphique
def save_frise_rttm(annotation_):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6),sharex=True)
    notebook.plot_annotation(annotation_,ax=axes)
    fig.savefig("figures/frise.png", bbox_inches="tight")
    plt.close(fig)#libère memory



def get_representative_sample(folder_path,wav_folder,sample_size,all=False,seed=130):
    """go through rttm files in a folder, add a sample of them to a dictionnaire of annotations,if all==True then go through all the files of the folder"""
    directory=Path(folder_path)
    file_statistics=[]#recup les stats des fichiers, pour un sample représentatif
    folder_annotations={}
    i=0#indice du fichier
    summation=0#compter le nombre de fichiers dans le path
    for file in sorted(directory.glob("*.rttm")):#sorted tri, glob vient recup tt les files .rttm
        summation+=1
        folder_annotations[file.stem]=load_rttm(file)[file.stem]#charger l'annotation dans le dico d'annotations
        annot_file=folder_annotations[file.stem]#alleger le code visuellement
        wav_file=Path(wav_folder)/f"{file.stem}.wav"

        #recup les stats de chaque fichier
        file_statistics.append([None]*5)
        file_statistics[i][0]=file.stem#nom du fichier pour l'indice i correspondant
        file_statistics[i][1]=len(annot_file.labels())#nombres de loctueurs
        file_statistics[i][2]=soundfile.info(wav_file).duration #size en secondes, l'indice est l'indice du fichier trié dans l'ordre alphabetique pendant la laecture du dossier
        file_statistics[i][3]=annot_file.get_overlap().duration()/annot_file.get_timeline().support().duration()#overlap en % de parole
        file_statistics[i][4]=annot_file.get_timeline().support().duration() / soundfile.info(wav_file).duration #pourcentage de parole du fichier
        i+=1
    if all==True:#if all folder desired
        return folder_annotations

    ####CALCUL DU SAMPLE####
    print(summation)#test, à supprimer
    sample_size=sample_size/summation#taille du sample en %

    #tri en fonction du nb de speaker
    file_statistics=sorted(file_statistics, key=lambda t: t[1])

    #divisions en batch represnetatifs
    batch_spk1 = file_statistics[: summation // 3]
    batch_spk2 = file_statistics[summation // 3 : 2 * summation // 3]
    batch_spk3 = file_statistics[2 * summation // 3 :]

    #tri de chaque batch en fonction de overlap
    batch_spk1=sorted(batch_spk1, key=lambda t: t[3])
    batch_spk2=sorted(batch_spk2, key=lambda t: t[3])
    batch_spk3=sorted(batch_spk3, key=lambda t: t[3])

    #subdivisions de chaque batch en sous batch
    #spk1
    batch_ov1 = batch_spk1[: len(batch_spk1) // 3]
    batch_ov2 = batch_spk1[len(batch_spk1) // 3 : 2 * len(batch_spk1) // 3]
    batch_ov3 = batch_spk1[2 * len(batch_spk1) // 3 :]
    #spk2
    batch_ov4 = batch_spk2[: len(batch_spk2) // 3]
    batch_ov5 = batch_spk2[len(batch_spk2) // 3 : 2 * len(batch_spk2) // 3]
    batch_ov6 = batch_spk2[2 * len(batch_spk2) // 3 :]
    #spk3
    batch_ov7 = batch_spk3[: len(batch_spk3) // 3]
    batch_ov8 = batch_spk3[len(batch_spk3) // 3 : 2 * len(batch_spk3) // 3]
    batch_ov9 = batch_spk3[2 * len(batch_spk3) // 3 :]

    #tri de chaque batch en fonction du poyurcentage de parole
    batch_ov1=sorted(batch_ov1, key=lambda t: t[4])
    batch_ov2=sorted(batch_ov2, key=lambda t: t[4])
    batch_ov3=sorted(batch_ov3, key=lambda t: t[4])
    batch_ov4=sorted(batch_ov4, key=lambda t: t[4])
    batch_ov5=sorted(batch_ov5, key=lambda t: t[4])
    batch_ov6=sorted(batch_ov6, key=lambda t: t[4])
    batch_ov7=sorted(batch_ov7, key=lambda t: t[4])
    batch_ov8=sorted(batch_ov8, key=lambda t: t[4])
    batch_ov9=sorted(batch_ov9, key=lambda t: t[4])

    ####division de chaque batch en 3 sous batch ####
    sub_batches=[]
    for batch in [batch_ov1,batch_ov2,batch_ov3,batch_ov4,batch_ov5,batch_ov6,batch_ov7,batch_ov8,batch_ov9]:
        n=len(batch)
        sub_batches.append(batch[: n // 3])
        sub_batches.append(batch[n // 3 : 2 * n // 3])
        sub_batches.append(batch[2 * n // 3 :])

    #### AJOUT : nombre de fichiers à tirer dans chaque sous batch ####
    target=round(sample_size*summation)#nombre total de fichiers voulus
    quotas=[len(sb)*sample_size for sb in sub_batches]#part "idéale" (psk float) de chaque sous batch
    counts=[int(q) for q in quotas]#partie entière, ce qu'on va recuperer comme fichier dans un batch k
    #répartir les fichiers restants aux sous batchs qui ont la plus grande partie décimale
    remaining=target-sum(counts)#ce qu'on doit encore rajouter
    order=sorted(range(len(quotas)), key=lambda k: quotas[k]-counts[k], reverse=True)#on trie selon la partie decimale de chaque sous batch ->quots avec float et coutns sans float, order contient les indices
    for k in order[:remaining]:
        counts[k]+=1#+1 un fichier à recup dans le batch d'indice k

    #### AJOUT : tirage aléatoire (graine fixe) dans chaque sous batch et fusion ####
    rng=random.Random(seed)
    sample_annotations={}
    for sb,k in zip(sub_batches,counts):#zip parcourt les deux listes en parallèle, case par case et merci claude
        for row in rng.sample(sb,k):#rng sample tire k lignes differentes, dans sb
            sample_annotations[row[0]]=folder_annotations[row[0]]#row[0] stem du fichier tiré
    return sample_annotations

##MADE BY CLAUDE
def save_sample_rttm(sample, output_path, split_files=False):
    """
    Save the annotations returned by get_representative_sample in RTTM format.

    Parameters
    ----------
    sample : dict[str, pyannote.core.Annotation]
        Output of get_representative_sample: {file_stem: Annotation}.
    output_path : str or Path
        If split_files is False: path of the single RTTM file to write
        (e.g. "data/samples/sample_dev_seed42.rttm").
        If split_files is True: folder in which one "<stem>.rttm" file
        per audio file is written.
    split_files : bool, default False
        False -> all annotations in one RTTM file (one block of lines per uri).
        True  -> one RTTM file per audio file.

    Returns
    -------
    list[Path]
        Paths of the RTTM files that were written.

    Notes
    -----
    - The uri (field 2 of each RTTM line) is forced to the file stem, so that
      load_rttm(path)[stem] works when the file is read back.
    - Annotations are copied before setting the uri: the input dict is not modified.
    - Files are written in sorted stem order, so the output is deterministic.
    - pyannote writes times with 3 decimals (1 ms precision).
    """
    output_path = Path(output_path)
    written_files = []

    if split_files:
        output_path.mkdir(parents=True, exist_ok=True)
        for stem in sorted(sample):
            annotation = sample[stem].copy()
            annotation.uri = stem
            file_path = output_path / f"{stem}.rttm"
            with open(file_path, "w") as f:
                annotation.write_rttm(f)
            written_files.append(file_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for stem in sorted(sample):
                annotation = sample[stem].copy()
                annotation.uri = stem
                annotation.write_rttm(f)
        written_files.append(output_path)

    return written_files


# ############TESTS BY CLAUDE###############

# RTTM_DIR = Path("data/voxconverse/annotations/dev")
# WAV_DIR = Path("data/voxconverse/dev/audio")
# OUT_CSV = Path("figures/sample_vs_dev_smd_ks.csv")
# SAMPLE_SIZE = 90

# SEED = 130
 
# # Variables utilisées par la stratification (équilibrées par construction)
# STRATIFIED = ["n_speakers", "overlap_rate", "speech_ratio"]
# # Variables NON utilisées par la stratification (le vrai test)
# OTHERS = ["duration", "n_segments", "mean_segment"]
 
# SMD_THRESHOLD = 0.1  # repère usuel : |SMD| < 0.1 = bon équilibre
 
 
# def compute_file_stats(stem, annotation, wav_dir=WAV_DIR):
#     """Statistiques d'un fichier, à partir de son Annotation de référence et de son .wav."""
#     duration = sf.info(str(wav_dir / f"{stem}.wav")).duration
#     speech = annotation.get_timeline().support().duration()
#     overlap = annotation.get_overlap().duration()
#     seg_durations = [seg.duration for seg, _ in annotation.itertracks()]
#     return {
#         "file": stem,
#         "n_speakers": len(annotation.labels()),
#         "overlap_rate": overlap / speech if speech > 0 else 0.0,
#         "speech_ratio": speech / duration,
#         "duration": duration,
#         "n_segments": len(seg_durations),
#         "mean_segment": float(np.mean(seg_durations)) if seg_durations else 0.0,
#     }
 
 
# def build_dev_table(rttm_dir=RTTM_DIR, wav_dir=WAV_DIR):
#     """DataFrame des stats de tous les fichiers du dev (une ligne par fichier)."""
#     rows = []
#     for path in sorted(rttm_dir.glob("*.rttm")):
#         annotation = next(iter(load_rttm(str(path)).values()))
#         rows.append(compute_file_stats(path.stem, annotation, wav_dir))
#     return pd.DataFrame(rows)
 
 
# def smd(sample_values, dev_values):
#     sd = dev_values.std(ddof=1)
#     return (sample_values.mean() - dev_values.mean()) / sd if sd > 0 else 0.0
 
 
# def compare_sample_to_dev(df_dev, sample_stems):
#     in_sample = df_dev["file"].isin(sample_stems)
#     s, comp = df_dev[in_sample], df_dev[~in_sample]
#     rows = []
#     for var in STRATIFIED + OTHERS:
#         ks_comp = ks_2samp(s[var], comp[var])
#         value = smd(s[var], df_dev[var])
#         rows.append({
#             "variable": var,
#             "stratifiee": var in STRATIFIED,
#             "mean_sample": s[var].mean(),
#             "mean_dev": df_dev[var].mean(),
#             "median_sample": s[var].median(),
#             "median_dev": df_dev[var].median(),
#             "SMD": value,
#             "SMD_ok": abs(value) < SMD_THRESHOLD,
#             "D_vs_dev": ks_2samp(s[var], df_dev[var]).statistic,
#             "D_vs_complement": ks_comp.statistic,
#             "p_vs_complement": ks_comp.pvalue,
#         })
#     return pd.DataFrame(rows)
 
 
# def main():
#     sample = get_representative_sample(str(RTTM_DIR), str(WAV_DIR), SAMPLE_SIZE, seed=SEED)
#     df_dev = build_dev_table()
#     result = compare_sample_to_dev(df_dev, list(sample.keys()))
 
#     print(f"Sample : {len(sample)} fichiers / dev : {len(df_dev)} fichiers (seed={SEED})\n")
#     print(result.round(3).to_string(index=False))
 
#     OUT_CSV.parent.mkdir(exist_ok=True)
#     result.to_csv(OUT_CSV, index=False)
#     print(f"\nTableau sauvegardé dans {OUT_CSV}")
 
 
# if __name__ == "__main__":
#     main()




# ###1st TESTS### BY G
# sample = get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", 40)
# print(len(sample))
# print(sample)

#tests
# test=load_file_annotations("zyffh.rttm")
# print(test)
# notebook.crop = Segment(t1, t2)
# save_frise_rttm(test)

#ecriture des samples en rttm

#save_sample_rttm(get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", 40),"samples/sample40_seed130.rttm")
#save_sample_rttm(get_representative_sample("data/voxconverse/annotations/dev", "data/voxconverse/dev/audio", 90),"samples/sample90_seed130.rttm")

save_sample_rttm(get_representative_sample("data/voxconverse/annotations/test", "data/voxconverse/test/voxconverse_test_wav", 40),"samples/test/sample40_seed130.rttm")
save_sample_rttm(get_representative_sample("data/voxconverse/annotations/test", "data/voxconverse/test/voxconverse_test_wav", 90),"samples/test/sample90_seed130.rttm")