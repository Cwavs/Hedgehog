from numpy import ndarray

#Create a base preprocessor class.
class _preprocessor():
    
    #Set creation parameters.
    def __init__(self, sampleRate: int) -> None:
        #Store values.
        self.sampleRate = sampleRate
    
    #Define an invocation function to be overridden later.
    def Invoke(self, audio: ndarray) -> ndarray:
        #Because this is the base preprocessor that isn't meant to be used. I'm just returning the audio raw here.
        return audio

#Create the neural pre processor class as an overide of the base preprocessor class.
class neuralPreProcessor(_preprocessor):

    #Set creation parameters. Including the defaults derived from corresponding values used in the original script.
    def __init__(self, sampleRate: int, features: int = 96, windowLength: int = 400, hopLength: int = 160, segmentLength: int = 187) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(sampleRate)

        #Store the rest of the values.
        self.features = features
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.segmentLength = segmentLength

    #Define our own invocation function.
    def Invoke(self, audio: ndarray) -> ndarray:
        #Import some libs needed for processing.
        from librosa.feature import melspectrogram
        from numpy import stack, newaxis

        #Generate a melspectrogram and transpose it using a combination of the parameters provided from the parent class and the parameters of this function.
        _mel = melspectrogram(
            #Data to be spectrogrammed.
            y=audio,
            #Samplerate of the provided audio data.
            sr=self.sampleRate,
            #We use the features parameter to define the number of frequency bins to extract.
            n_mels=self.features,
            #Fairly obvious, we set the hopLength and windowLength to their respective parameters, and match n_fft with windowLength.
            hop_length=self.hopLength,
            win_length=self.windowLength,
            n_fft=self.windowLength
        ).T

        #Here we split up the mel spectrogram into slice/segments as per the provided segmentLength.
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

        #We then return these segments to be processed by the model.
        return _segments

class experimentalNeuralPreProcessor(neuralPreProcessor):

    #Set creation parameters. Including the defaults derived from corresponding values used in the original script.
    def __init__(self, sampleRate: int, features: int = 96, windowLength: int = 400, hopLength: int = 160, segmentLength: int = 187) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(sampleRate, features, windowLength, hopLength, segmentLength)

    #Define our own invocation function.
    def Invoke(self, audio: ndarray) -> ndarray:
        #Import some libs needed for processing.
        from librosa.feature import melspectrogram
        from essentia import log
        from essentia.standard import Windowing, Spectrum, MelBands, FrameGenerator, TensorflowInputMusiCNN
        from essentia import Pool
        from numpy import stack, newaxis, empty
        from pyloudnorm.meter import Meter
        from pyloudnorm.normalize import loudness

        log.infoActive = False
        log.warningActive = False
        log.errorActive = False

        #audio = loudness(audio, Meter(self.sampleRate).integrated_loudness(audio), -14)

        _segments = Pool()

        for frame in FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True, lastFrameToEndOfFile=True, validFrameThresholdRatio=1):
            fMel = TensorflowInputMusiCNN()(frame)
            _segments.add("mel96", fMel)

        _mel = _segments["mel96"]
        print(_mel.shape)

        '''#Here we split up the mel spectrogram into slice/segments as per the provided segmentLength.
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
            _segments = _segments[0][newaxis, ...]'''

        #We then return these segments to be processed by the model.
        return _mel

#Create the traditional pre processor class as an overide of the base preprocessor class.
class traditionalPreProcessor(_preprocessor):

    #Set creation parameters. Including the defaults derived from corresponding values used in the original script.
    def __init__(self, sampleRate: int = 22050, features: int = 4, windowLength: int = 1024, hopLength: int = 512) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(sampleRate)

        #Store the rest of the values.
        self.features = features
        self.winLenth = windowLength
        self.hopLength = hopLength

    #Define our own invocation function.
    def Invoke(self, audio: ndarray) -> ndarray:
        #Import some libs needed for processing.
        from librosa.feature import mfcc
        from numpy import random, mean, cov

        #Obtain the mfccs for the audio data.
        mel = mfcc(
            #Data to use.
            y=audio,
            sr=self.sampleRate,
            #Number of mfccs to generate (aka features to extract).
            n_mfcc=self.features,
            #I got these values from Musly. Thanks Musly.
            hop_length=self.hopLength,
            win_length=self.winLenth,
            n_fft=self.winLenth
        )

        #Extract the most "statistically significant" values from the music (I am bad at maths, that might be a bad explination). Idea taken from the Musly paper but it does seem to do a good job of distilling the song down to the needed parts.
        return random.default_rng().multivariate_normal(mean(mel, axis=1), cov(mel), size=(4, 4))