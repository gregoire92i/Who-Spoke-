import librosa, librosa.display, soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pyannote.database.util import load_rttm


#parameters
sr=16000#frequence d'echantillonage
frame_size=64
hop_size=10


#charger la data
def load_audio (path,sr):
    y,sr=librosa.load(path,sr=16000,mono=True)
    return y

#calcul fft
def FFT(signal_audio,frame_size=frame_size,hop_size=hop_size):
    """#fait la tranformée de fourier rapide du signal, paramètres en ms"""
    signal_audio=signal_audio#car stft prend en parametres le nombre d'echantillons par trame
    frame_size=frame_size*16
    hop_size=hop_size*16
    hann_size=frame_size# on veut que recouvrir exactement la frame
    FFT_output=librosa.stft(signal_audio,n_fft=frame_size,hop_length=hop_size,win_length=hann_size)
    return FFT_output


# #echantillonage
# audio=load_audio("data/test.wav", sr)

# print("forme :", audio.shape, "| durée :", len(audio) / sr, "s | min/max :", audio.min(), audio.max())
# print(sr)
# print(len(audio) / sr)        
# print(librosa.get_duration(y=audio, sr=sr))

# #graphique
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6),sharex=True)# nrows le nombre de ligne ncols colonne donc -> nbr de graphiques dans la figure=nrows*ncols
# librosa.display.waveshow(audio,sr=sr,ax=axes[0])
# axes[0].set(title="Forme d'onde")
# plt.savefig("figures/waveform.png")

# #trames et FFT
# D=FFT(audio)
# print(D.shape)

# #spectrogramme
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)#amplitude to db convertit en db, np.abs prend le module de D (complexe->reel), ref est valeur de ref (à l'origine ?)np.max prend la val max du spectrogramme --> important pour comparer diff fichiers entre eux qui n'ont pas dutout le meme volume

# #graphique
# spectro = librosa.display.specshow(S_db, sr=sr, hop_length=hop_size*16, x_axis="time", y_axis="hz", ax=axes[1])
# fig.colorbar(spectro, ax=axes[1], format="%+2.0f dB")
# axes[1].set(title="Spectrogramme")
# plt.tight_layout()#ajuste les marges pour eviter les chevauchements
# plt.savefig("figures/waveform.png")


# ######################
# #FICHIER2#

# #echantillonage
# audio1=load_audio("data/enregistrement.wav", sr)

# #graphique
# fig1, axes1 = plt.subplots(nrows=2, ncols=1, figsize=(12, 6),sharex=True)# nrows le nombre de ligne ncols colonne donc -> nbr de graphiques dans la figure=nrows*ncols
# librosa.display.waveshow(audio1,sr=sr,ax=axes1[0])
# axes1[0].set(title="Forme d'onde")

# #trames et FFT
# D1=FFT(audio1)
# print(D1.shape)

# #spectrogramme
# S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)#amplitude to db convertit en db, np.abs prend le module de D (complexe->reel), ref est valeur de ref (à l'origine ?)np.max prend la val max du spectrogramme --> important pour comparer diff fichiers entre eux qui n'ont pas dutout le meme volume

# #graphique
# spectro1 = librosa.display.specshow(S_db1, sr=sr, hop_length=hop_size*16, x_axis="time", y_axis="hz", ax=axes1[1])
# fig1.colorbar(spectro1, ax=axes1[1], format="%+2.0f dB")
# axes1[1].set(title="Spectrogramme")
# plt.tight_layout()#ajuste les marges pour eviter les chevauchements
# plt.savefig("figures/waveform1.png")


# annot=load_rttm("data/voxconverse/annotations/dev/abjxc.rttm")["abjxc"]
# print(annot)
# print(annot.label_duration("spk00"))

