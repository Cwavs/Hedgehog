from librosa import load, power_to_db
from librosa.feature import melspectrogram
from numpy import amax, stack, newaxis, ndarray

class _preprocessor():
    def __init__(self, audio: ndarray) -> None:
        self.audio = audio
    
    def Invoke(self):
        pass