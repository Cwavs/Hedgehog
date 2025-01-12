from argparse import ArgumentParser
from librosa import load
from librosa.feature import mfcc
from numpy import savetxt, random, mean, cov
from pathlib import Path

#Setup argparser.
parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

#Set up arugments to be parsed.
parser.add_argument('audioDir', help="Root directory of music library.", type=Path)
parser.add_argument('-c', '--csvDir', help="Directory to save the csv files to.", type=Path, default=None)
parser.add_argument("-f", "--format", help="File extension/format of the audio files to read.", type=str, default="flac")

#Parse args.
args = parser.parse_args()

#Create an instance of numpys rng system.
rng = random.default_rng()

#Obtain list of audio files and loop through them.
for file in args.audioDir.rglob(f"*.{args.format}"):

    #Print the current file being processed.
    print(file)

    #Check if a csv file already exists. If it does we skip it rather than regenerating it.
    #This is hacky and looks super fucking cursed. It works but I sure hope someone else comes along with a better way.
    if Path(str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv").is_file() == False:

        #Load up the audio file.
        audioData = load(file)[0]

        #Obtain the mfccs for the audio data.
        mel = mfcc(
            #Data to use
            y=audioData,
            #Number of mfccs to generate (aka features to extract)
            n_mfcc=4,
            #I got these values from Musly. Thanks Musly.
            hop_length=512,
            win_length=1024,
            n_fft=1024
        )

        #Extract the most "statistically significant" values from the music (I am bad at maths, that might be a bad explination). Idea taken from the Musly paper but it does seem to do a good job of distilling the song down to the needed parts.
        normal = rng.multivariate_normal(mean(mel, axis=1), cov(mel), size=(4, 4))

        #Reshape the array to be one dimensional (ala one string of 64 numbers) and save it to a text file.
        savetxt(
            str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
            normal.reshape(-1),
            delimiter=","
        )
    else:
        print("csv file already exists for " + file.name + ", assuming already fingerprinted and skipping.")