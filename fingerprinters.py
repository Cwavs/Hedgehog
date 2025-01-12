from numpy import ndarray 
from preprocessors import _preprocessor
from pathlib import Path
from tensorflow import lite

class _fingerprinter:

    def __init__(self, preprocessor: _preprocessor, audio: ndarray) -> None:
        self.preprocessor = preprocessor
        self.audio = audio

    def Invoke(self) -> ndarray:
        return self.Preprocess(self)

    def Preprocess(self) -> ndarray:
        return self.preprocessor.Invoke(self.audio)
    
class neuralFingerprinter(_fingerprinter):
    
    def __init__(self, preprocessor: _preprocessor, audio: ndarray, model: Path) -> None:
        super().__init__(preprocessor, audio)
        self.model = Path(model)

    def Invoke(self) -> ndarray:

        #Setup and load Tensorflow model.
        _interpreter = lite.Interpreter(model_path=self.model.as_posix())

        #Get input and output details
        inputDetails = _interpreter.get_input_details()[0]
        outputDetails = _interpreter.get_output_details()[0]

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