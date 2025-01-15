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
    
    #Set creation parameters. We add an aditional model parameter here so that we can accept a tflite model of out choosing.
    def __init__(self, preprocessor: _preprocessor, audio: ndarray, model: Path) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(preprocessor, audio)

        #Store the rest of the values.
        self.model = Path(model)

    #Define our own invocation function.
    def Invoke(self) -> ndarray:
        #Import tensorflow for our neural work.
        from tensorflow import lite

        #Setup and load Tensorflow model.
        _interpreter = lite.Interpreter(model_path=self.model.as_posix())

        #Get input and output details
        inputDetails = _interpreter.get_input_details()[0]
        outputDetails = _interpreter.get_output_details()[0]

        #Call the selected preprocessor and get the result.
        _preprocessed = self.Preprocess()

        #Resize the input array to take the newly stacked array. TFlite does not support dynamic inputs, so we have to explictly resize this to match the input array.
        _interpreter.resize_tensor_input(0, _preprocessed.shape)
        _interpreter.allocate_tensors()

        #Input the result into the model.
        _interpreter.set_tensor(inputDetails['index'], _preprocessed)

        #Actually run the model.
        _interpreter.invoke()

        #Get the output data from the model.
        outputData = _interpreter.get_tensor(outputDetails['index'])

        #Because we've passed n segements in an early step, the output will be (n,50), as such we have to take the mean of the array to get the average fingerprint of the song.
        return outputData.mean(axis=0)

#This is an experimental fingerprinter made so that I can test new approaches on the model without impacting the normal function when I push changes to bugfix.
class experimentalNeuralFingerprinter(neuralFingerprinter):
    
    #Set creation parameters. Including the defaults derived from corresponding values used in the original script.
    def __init__(self, preprocessor: _preprocessor, audio: ndarray, model: Path) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(preprocessor, audio, model)

        #Define our own invocation function.
    def Invoke(self) -> ndarray:
        #Import tensorflow for our neural work.
        from tensorflow import lite
        from numpy import average, sqrt

        #Setup and load Tensorflow model.
        _interpreter = lite.Interpreter(model_path=self.model.as_posix())

        #Get input and output details
        inputDetails = _interpreter.get_input_details()[0]
        outputDetails = _interpreter.get_output_details()[0]

        #Call the selected preprocessor and get the result.
        _preprocessed = self.Preprocess()

        #Resize the input array to take the newly stacked array. TFlite does not support dynamic inputs, so we have to explictly resize this to match the input array.
        _interpreter.resize_tensor_input(0, _preprocessed.shape)
        _interpreter.allocate_tensors()

        #Input the result into the model.
        _interpreter.set_tensor(inputDetails['index'], _preprocessed)

        #Actually run the model.
        _interpreter.invoke()

        #Get the output data from the model.
        outputData = _interpreter.get_tensor(outputDetails['index'])

        rms = sqrt((outputData**2).mean(axis=1))

        #Because we've passed n segements in an early step, the output will be (n,50), as such we have to take the mean of the array to get the average fingerprint of the song.
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