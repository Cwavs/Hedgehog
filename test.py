from fingerprinters import neuralFingerprinter
from preprocessors import neuralPreProcessor
from librosa import load

audio, sr = load("/home/cwavs/Documents/Hedgehog/Music/Chonny Jash/Chonny's Charming Chaos Compendium, Vol․ 1/01 Time Machine Reprise.flac", mono=True, sr=16000)

preprocessor = neuralPreProcessor(sr=sr, features=96, winLength=400, hopLength=160, segmentLength=400)
fingperinter = neuralFingerprinter(preprocessor=preprocessor, audio=audio, model="/home/cwavs/Documents/Hedgehog/Music.tflite")

print(fingperinter.Invoke())