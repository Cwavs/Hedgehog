from fingerprinters import traditionalFingerprinter
from preprocessors import traditionalPreProcess
from librosa import load

audio, sr = load("/home/cwavs/Documents/Hedgehog/Music/Chonny Jash/Chonny's Charming Chaos Compendium, Vol․ 1/01 Time Machine Reprise.flac")

preprocessor = traditionalPreProcess(sr=sr)
fingperinter = traditionalFingerprinter(preprocessor=preprocessor, audio=audio)

print(fingperinter.Invoke())