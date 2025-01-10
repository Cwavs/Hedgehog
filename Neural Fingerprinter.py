from argparse import ArgumentParser
from librosa import load, power_to_db
from librosa.feature import melspectrogram
from numpy import savetxt, amax, stack, newaxis
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

#Get input and output details
inputDetails = interpreter.get_input_details()[0]
outputDetails = interpreter.get_output_details()[0]

#Obtain list of audio files and loop through them
for file in args.audioDir.rglob(f"*.{args.format}"):

    #Print the current file being processed
    print(file)

    #Load audio file in stereo
    audioData, sr = load(file, mono=True, sr=16000)

    #Obtain the mfcc for the audio data
    mel = melspectrogram(
        y=audioData,
        sr=sr,
        #I chose 96 mels because it seemed like the more likely of the two axes of the input array to be used for this, could be wrong, but it seems to work
        n_mels=96,
        #I borrowed the hop length and window length from Musly, again could be wrong, but it seems to work, thanks Musly
        hop_length=160,
        win_length=400,
        n_fft=400
    )

    mel = power_to_db(mel, ref=amax).T

    print(mel.shape)

    segs = []
    start = 0
    if mel.shape[0] > 187:
        while start+187 <= mel.shape[0]:
            segs.append(mel[start:start+187, :])
            start += 187
    else:
        segs.append(mel)
    
    if(len(segs) != 1):
        segs = stack(segs, axis=0)
    else:
        segs = segs[0][newaxis, ...]

    #Resize the input array to take a stereo file
    interpreter.resize_tensor_input(0, segs.shape)
    interpreter.allocate_tensors()

    #Input the result into the model
    interpreter.set_tensor(inputDetails['index'], segs)

    #Actually run the model
    interpreter.invoke()

    #Get the output data from the model
    outputData = interpreter.get_tensor(outputDetails['index'])

    print(outputData.shape)

    outputData = outputData.mean(axis=0)

    print(outputData.shape)
    print(outputData)

    #Reshape the data into a 1-D array and save it to a csv
    savetxt(
        str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
        outputData.reshape(-1),
        delimiter=","
    )