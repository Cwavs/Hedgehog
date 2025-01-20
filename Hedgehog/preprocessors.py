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