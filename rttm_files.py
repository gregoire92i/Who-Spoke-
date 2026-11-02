from pyannote.core import notebook  #pr structure des données (segment, annotations, timeline)
from pyannote.database.util import load_rttm #chargement des fichiers 
from pathlib import Path #pour manipuler les fichiers (supprimer les char du type (.pdf), etc...)
import numpy as np
import matplotlib.pyplot as plt#affichage

def load_file_annotations(rttm_file):
    file=Path(rttm_file)
    annotation=load_rttm(Path("data/voxconverse/annotations/dev") / file)[file.stem]
    return annotation


#graphique
def save_frise_rttm(annotation_):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6),sharex=True)
    notebook.plot_annotation(annotation_,ax=axes)
    fig.savefig("figures/frise.png", bbox_inches="tight")
    plt.close(fig)#libère memory

def get_representative_files(annotations_,folder_path,all=False):
    """go through rttm files in a folder, add them to a dictionnaire annotations_,if all==True then go through alle the files of the folder"""
    directory=Path(folder_path)
    for file in sorted(directory.glob("*.rttm")):#sorted tri, glob vient recup tt les files .rttm
        annotations_[file.stem]=



#tests
# test=load_file_annotations("zyffh.rttm")
# print(test)
# notebook.crop = Segment(t1, t2)
# save_frise_rttm(test)