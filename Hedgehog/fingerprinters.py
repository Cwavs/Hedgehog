from numpy import ndarray
from preprocessors import _preprocessor
from preprocessors import _preprocessor
from pathlib import Path

#Create a base fingerprinter class.
class _fingerprinter:

    #Set creation parameters.
    def __init__(self, preprocessor: _preprocessor, audio: ndarray) -> None:
        #Store values.
        self.preprocessor = preprocessor
        self.audio = audio
    
    #Define an invocation function to be overridden later.
    def Invoke(self) -> ndarray:
        #Because this is the base fingerprinter that isn't meant to be used. I'm just returning the output of the preprocessor here.
        return self.Preprocess(self)

    #Define a preprocess function to be overridden later.
    def Preprocess(self) -> ndarray:
        #Here I'm just calling and returning the result of the preprocessor. I can't really think of a reason I'd need to process this more here, so I'm just going to inherit this in the following classes.
        return self.preprocessor.Invoke(self.audio)

#Create the neural fingerprinter class as an overide of the base fingerprinter class.
class neuralFingerprinter(_fingerprinter):
    
    #Set creation parameters.
    def __init__(self, preprocessor: _preprocessor, audio: ndarray, model: Path) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(preprocessor, audio)
        self.model = model

        #Define our own invocation function.
    def Invoke(self) -> ndarray:
        #Essentia's prebuilt function for MusiCNN predictions.
        from essentia.standard import TensorflowPredictMusiCNN
        #Import Essentia's logging class.
        from essentia import log
        #Import Numpy average and square root functions for RMS (Root Means Square) Weighted Average.
        from numpy import average, sqrt

        #Disable Essentia Logging.
        log.infoActive = False
        log.warningActive = False
        log.errorActive = False

        #Get the output data from the model.
        outputData = TensorflowPredictMusiCNN(graphFilename=self.model.as_posix(), lastPatchMode="repeat")(self.audio)

        #Create the RMS weights.
        rms = sqrt((outputData**2).mean(axis=1))

        #Average together the different segments with the RMS weights. Then return it.
        return average(outputData, axis=0, weights=rms)

#Create the traditional fingerprinter class as an overide of the base fingerprinter class.
class traditionalFingerprinter(_fingerprinter):

    #Set creation parameters.
    def __init__(self, preprocessor: _preprocessor, audio: ndarray) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(preprocessor, audio)
    
    #Define our own invocation function.
    def Invoke(self) -> ndarray:   
        #Return the result of the preprocessor.
        return self.Preprocess()