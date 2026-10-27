import librosa
import numpy as np
sr=16000#frequence d'echantillonage

#charger la data
def load_audio (path,sr):
    y,sr=librosa.load(path,sr=16000,mono=True)
    return y

#calcul fft
def FFT(signal_audio,frame_size,hop_size):
    """fait la tranformée de fourier rapide du signal, paramètres en ms"""
    signal_audio=signal_audio*16
    frame_size=frame_size*16
    hop_size=hop_size*16
    hann_size=frame_size# on veut que recouvrir exactement la frame
    FFT_output=librosa.stft(signal_audio,n_fft=frame_size,hop_length=hop_size,win_length=hann_size)
    return FFT_output



audio=load_audio("data/test.wav", sr)
print(audio.shape, audio.dtype)             
print(sr)
D=FFT(audio,40,20)

print(len(audio) / sr)                  
print(librosa.get_duration(y=audio, sr=sr))
print(D.shape)