from librosa import load, power_to_db
from librosa.feature import melspectrogram
from numpy import amax, stack, newaxis, ndarray

class _preprocessor():

    def __init__(self, sr: int) -> None:
        self.sr = sr
    
    def Invoke(self, audio: ndarray):
        return self.audio

class neuralPreProcessor(_preprocessor):

    def __init__(self, sr: int, features: int, winLength: int, hopLength: int, segmentLength: int) -> None:
        super.__init__(self, sr)
        self.features = features
        self.winLength = winLength
        self.hopLength = hopLength
        self.segmentLength = segmentLength

    def Invoke(self, audio: ndarray) -> ndarray:
        _mel = melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.features,
            hop_length=self.hopLength,
            win_length=self.winLength,
            n_fft=self.winLength
        )

        _mel = power_to_db(_mel, ref=amax).T

        _segments = []
        start = 0
        #We do this by first checking if there are enough samples to have more than one.
        if _mel.shape[0] > 187:
            #If there are we loop through, splicing it along the way and appending the segments to a list that we can use to stack them together later.
            while start+187 <= _mel.shape[0]:
                _segments.append(_mel[start:start+187, :])
                start += 187
        else:
            #If there aren't we just simply append the whole array to the list.
            _segments.append(_mel)
        
        #We then check the number of segments and decide whether to stack the arrays together or simply add a new dim. It's important this is done seperatley from the above step, as you cannot iteratively stack an array because numpy requires all arrays be the same shape in order to be stacked.
        if(len(_segments) != 1):
            _segments = stack(_segments, axis=0)
        else:
            _segments = _segments[0][newaxis, ...]

        return _segments