from argparse import ArgumentParser
from librosa import load, power_to_db
from librosa.feature import melspectrogram
from numpy import savetxt, amax, stack, newaxis
from pathlib import Path
from tensorflow import lite

#Set up argparser.
parser = ArgumentParser(prog="Hedgehog Neural", description="Uses AI to analyse audio tracks.")

#Set up arugments to be parsed.
parser.add_argument('audioDir', help="Root directory of music library.", type=Path)
parser.add_argument('-c', '--csvDir', help="Directory to save the csv files to.", type=Path, default=None)
parser.add_argument("-m", "--model", help="Path to model file.", default="./Music.tflite", type=Path)
parser.add_argument("-f", "--format", help="File extension/format of the audio files to read.", type=str, default="flac")

#Parse args.
args = parser.parse_args()

#Setup and load Tensorflow model.
interpreter = lite.Interpreter(model_path=args.model.as_posix())

#Get input and output details
inputDetails = interpreter.get_input_details()[0]
outputDetails = interpreter.get_output_details()[0]

#Obtain list of audio files and loop through them.
for file in args.audioDir.rglob(f"*.{args.format}"):

    #Print the current file being processed.
    print(file)
    
    #Check if a csv file already exists. If it does we skip it rather than regenerating it.
    #This is hacky and looks super fucking cursed. It works but I sure hope someone else comes along with a better way.
    if Path((str(file).rsplit('.', 1)[0]) + ".csv").is_file() == False:

        #Load audio file in mono at 16khz.
        audioData, sr = load(file, mono=True, sr=16000)

        #Obtain the melspectrogram for the audio data.
        mel = melspectrogram(
            #Data to be spectrogrammed.
            y=audioData,
            #Sample rate we're working with.
            sr=sr,
            #I chose 96 mels because it seemed like the more likely of the two axes of the input array to be used for this, could be wrong, but it seems to work.
            n_mels=96,
            #These align with the 187 in the input shape (when at 16khz at least), which I've assumed is the slice of audio it works with.
            hop_length=160,
            win_length=400,
            n_fft=400
        )

        #Switch from power to decibels, and grab the transposed version so that the dimensions align with those used in the model's input shape.
        mel = power_to_db(mel, ref=amax).T

        #Here we split up the mel spectrogram into slice/segments of 187 samples.
        segs = []
        start = 0
        #We do this by first checking if there are enough samples to have more than one.
        if mel.shape[0] > 187:
            #If there are we loop through, splicing it along the way and appending the segments to a list that we can use to stack them together later.
            while start+187 <= mel.shape[0]:
                segs.append(mel[start:start+187, :])
                start += 187
        else:
            #If there aren't we just simply append the whole array to the list.
            segs.append(mel)
        
        #We then check the number of segments and decide whether to stack the arrays together or simply add a new dim. It's important this is done seperatley from the above step, as you cannot iteratively stack an array because numpy requires all arrays be the same shape in order to be stacked.
        if(len(segs) != 1):
            segs = stack(segs, axis=0)
        else:
            segs = segs[0][newaxis, ...]

        #Resize the input array to take the newly stacked array. TFlite does not support dynamic inputs, so we have to explictly resize this to match the input array.
        interpreter.resize_tensor_input(0, segs.shape)
        interpreter.allocate_tensors()

        #Input the result into the model.
        interpreter.set_tensor(inputDetails['index'], segs)

        #Actually run the model.
        interpreter.invoke()

        #Get the output data from the model.
        outputData = interpreter.get_tensor(outputDetails['index'])

        #Because we've passed n segements in an early step, the output will be (n,50), as such we have to take the mean of the array to get the average fingerprint of the song.
        outputData = outputData.mean(axis=0)

        #Save the data into a csv file in the specify dir.
        savetxt(
            str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
            outputData.reshape(-1),
            delimiter=","
        )
    else:
        print("csv file already exists for " + file.name + ", assuming already fingerprinted and skipping.")