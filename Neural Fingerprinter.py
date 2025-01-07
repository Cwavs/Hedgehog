from argparse import ArgumentParser
from librosa import load
from librosa.feature import mfcc
from numpy import savetxt, random, mean, cov, moveaxis, dtypes
from pathlib import Path
from tensorflow import lite

#Set up argparser
parser = ArgumentParser(prog="Hedgehog Neural", description="Uses AI to analyse audio tracks.")

#Set up arugments to be parsed
parser.add_argument('audioDir', help="Root directory of music library.", type=Path)
parser.add_argument('-c', '--csvDir', help="Directory to save the csv files to.", type=Path, default=None)
parser.add_argument("-m", "--model", help="Path to model file.", default="./Music.tflite")
parser.add_argument("-f", "--format", help="File extension/format of the audio files to read.", default="flac")

#Parse args
args = parser.parse_args()

#Setup and load Tensorflow model
interpreter = lite.Interpreter(model_path=args.model)

#Resize the input array to take a stereo file
interpreter.resize_tensor_input(0, (2, 187, 96))
interpreter.allocate_tensors()

#Get input and output details
inputDetails = interpreter.get_input_details()[0]
outputDetails = interpreter.get_output_details()[0]

#Obtain list of audio files and loop through them
for file in args.audioDir.rglob(f"*.{args.format}"):

    #Print the current file being processed
    print(file)

    #Load audio file in stereo
    audioData = load(file, mono=False)[0]

    #Obtain the mfcc for the audio data
    mel = mfcc(
        y=audioData,
        #I chose 96 mels because it seemed like the more likely of the two axes of the input array to be used for this, could be wrong, but it seems to work
        n_mfcc=96,
        #I borrowed the hop length and window length from Musly, again could be wrong, but it seems to work, thanks Musly
        hop_length=512,
        win_length=1024
    )

    #Calculate the multivariate normal of the mfcc
    normal = moveaxis(
        random.default_rng().multivariate_normal(
            #Calculate the mean value of the audio data
            mean(mel, axis=1),
            #Calculate the covariances of the audio data
            cov(mel), 
            size=(187, 96)
        ).astype(dtypes.Float32DType), [1, 2], [-1, 0]
    )

    #Input the result into the model
    interpreter.set_tensor(inputDetails['index'], normal)

    #Actually run the model
    interpreter.invoke()

    #Get the output data from the model
    outputData = interpreter.get_tensor(outputDetails['index'])

    #Reshape the data into a 1-D array and save it to a csv
    savetxt(
        str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
        outputData.reshape(-1),
        delimiter=","
    )