from numpy import ndarray

class _preprocessor():

    def __init__(self, sr: int) -> None:
        self.sr = sr
    
    def Invoke(self, audio: ndarray):
        return audio

class neuralPreProcessor(_preprocessor):

    def __init__(self, sr: int, features: int = 96, winLength: int = 400, hopLength: int = 160, segmentLength: int = 187) -> None:
        super().__init__(sr)
        self.features = features
        self.winLength = winLength
        self.hopLength = hopLength
        self.segmentLength = segmentLength

    def Invoke(self, audio: ndarray) -> ndarray:
        from librosa.feature import melspectrogram
        from librosa import power_to_db
        from numpy import amax, stack, newaxis

        _mel = melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.features,
            hop_length=self.hopLength,
            win_length=self.winLength,
            n_fft=self.winLength
        ).T

        _segments = []
        start = 0
        #We do this by first checking if there are enough samples to have more than one.
        if _mel.shape[0] > self.segmentLength:
            #If there are we loop through, splicing it along the way and appending the segments to a list that we can use to stack them together later.
            while start+187 <= _mel.shape[0]:
                _segments.append(_mel[start:start+self.segmentLength, :])
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

class traditionalPreProcessor(_preprocessor):
    def __init__(self, sr: int, features: int = 4, winLength: int = 1024, hopLength: int = 512) -> None:
        super().__init__(sr)
        self.features = features
        self.winLenth = winLength
        self.hopLength = hopLength

    def Invoke(self, audio: ndarray) -> ndarray:
        from librosa.feature import mfcc
        from numpy import random, mean, cov

        #Obtain the mfccs for the audio data.
        mel = mfcc(
            #Data to use
            y=audio,
            #Number of mfccs to generate (aka features to extract)
            n_mfcc=self.features,
            #I got these values from Musly. Thanks Musly.
            hop_length=self.hopLength,
            win_length=self.winLenth,
            n_fft=self.winLenth
        )

        #Extract the most "statistically significant" values from the music (I am bad at maths, that might be a bad explination). Idea taken from the Musly paper but it does seem to do a good job of distilling the song down to the needed parts.
        return random.default_rng().multivariate_normal(mean(mel, axis=1), cov(mel), size=(4, 4))