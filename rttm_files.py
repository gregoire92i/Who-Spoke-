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

def get_representative_sample(folder_path,sample_size,all=False):
    """go through rttm files in a folder, add a sample of them (64,128 or all) to a dictionnaire of annotations,if all==True then go through all the files of the folder"""
    directory=Path(folder_path)
    file_statistics[][]=0#recup les stats des fichiers, pour un sample représentatif
    i=0#indice du fichier
    summation=0#compter le nombre de fichiers dans le path
    for file in sorted(directory.glob("*.rttm")):#sorted tri, glob vient recup tt les files .rttm
        summation+=1
        folder_annotations[file.stem]=load_rttm(file)[file.stem]#charger l'annotation dans le dico d'annotations
        annot_file=folder_annotations[file.stem]#alleger le code visuellement

        #recup les stats de chaque fichier
        file_statistics[i][0]=file.stem#nom du fichier pour l'indice i correspondant
        file_statistics[i][1]=len(annot_file.labels())#nombres de loctueurs
        file_statistics[i][2]=soundfile.info(file).duration #size en secondes, l'indice est l'indice du fichier trié dans l'ordre alphabetique pendant la laecture du dossier
        file_statistics[i][3]=annot_file.get_overlap().duration()/annot_file.get_timeline().support().duration()#overlap en % de parole
        file_statistics[i][4]=annot_file.get_timeline().support().duration() / soundfile.info(file).duration #pourcentage de parole du fichier
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
    batch_ov1 = batch_spk1[: summation // 3]
    batch_ov2 = batch_spk1[summation // 3 : 2 * summation // 3]
    batch_ov3 = batch_spk1[2 * summation // 3 :]
    #spk2
    batch_ov4 = batch_spk2[: sum // 3]
    batch_ov5 = batch_spk2[sum // 3 : 2 * sum // 3]
    batch_ov6 = batch_spk2[2 * sum // 3 :]
    #spk3
    batch_ov7 = batch_spk3[: sum // 3]
    batch_ov8 = batch_spk3[sum // 3 : 2 * sum // 3]
    batch_ov9 = batch_spk3[2 * sum // 3 :]

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

    





#tests
# test=load_file_annotations("zyffh.rttm")
# print(test)
# notebook.crop = Segment(t1, t2)
# save_frise_rttm(test)