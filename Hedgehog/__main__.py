from fingerprinters import traditionalFingerprinter, neuralFingerprinter
from preprocessors import traditionalPreProcessor
from searchers import voyager, annoy
from voyager import Space
from argparse import ArgumentParser
from pathlib import Path
from librosa import load
from numpy import savetxt, ndarray, loadtxt
import os

#Disable Tensorflow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Define a function to find all the audio files in a dictionary.
def getAudioFiles(dir: Path, csvdir: Path, format: str) -> list:
    #Create an empty list of files.
    files = list()

    #Loop through the files in the provided dir of the provided format.
    for file in dir.rglob(f"*.{format}"):
        #If no csv exists in csvdir (or the music dir if there isn't a csv dir) then append the file to the list.
        if Path(str(file.with_suffix(".csv")) if not csvdir else f"{csvdir / (file.name.rsplit('.', 1)[0])}.csv").is_file() == False:
            files.append(file)
        #If a csv does exist, we skip the file and print a message about it to the user.
        else:
            print("Detected CSV file for " + str(file) + ", assuming already fingerprinted and skipping.")
    
    #Return the list.
    return files

#Define a function to save CSV files to the csv dir.
def saveCSVFile(file: Path, csvdir: Path, fingerprint: ndarray):
    #Save the provided fingerprint to the csv dir provided with the name of the file provided (with .csv instead of the original extension).
    savetxt(
        str(file.with_suffix(".csv")) if not csvdir else f"{csvdir / (file.name.rsplit('.', 1)[0])}.csv",
        fingerprint,
        fmt="%.18f",
        delimiter=","
    )

#Define a function to load CSV files from a given dir.
def loadCSVFiles(dir: Path) -> tuple:
    fingerprints = list()
    names = list()

    #Loop through files at the given dir with the csv extension
    for file in dir.rglob("*.csv"):
        fingerprints.append(loadtxt(file, delimiter=","))
        names.append(file.name)
    
    return (fingerprints, names)

#The function called if traditional fingerprinting is chosen.
def tradFingerprint(args):
    #Get a list of all the audio files at audioDir ignoring any with existing CSVs.
    files = getAudioFiles(args.audioDir, args.csvDir, args.format)

    #Loop through the files we found.
    for file in files:
        #Load the file with librosa.
        print("Trying to load " + str(file))
        audioData, sampleRate = load(file)
        #Create a new traditional pre-processor.
        preProcessor = traditionalPreProcessor(sampleRate)
        print("Currently Fingerprinting " + file.name)
        #Create and invoke the traditional fingerprinter.
        fingerprint = traditionalFingerprinter(preProcessor, audioData).Invoke()
        #Save the fingerprint to a csv file.
        saveCSVFile(file, args.csvDir, fingerprint)
    
    print("Done!")

#The function called if neural fingerprinting is chosen.
def neuralFingerprint(args):
    #Get a list of all the audio files at audioDir ignoring any with existing CSVs.
    files = getAudioFiles(args.audioDir, args.csvDir, args.format)

    #Loop through the files we found.
    for file in files:
        print("Trying to load " + str(file))
        #Load the file in mono at a sample rate of 16kHz with librosa.
        audioData = load(file, sr=16000, mono=True)[0]
        print("Currently Fingerprinting " + file.name)
        #Create and invoke the neural fingerprinter.
        fingerprint = neuralFingerprinter(None, audioData, args.model).Invoke()
        #Save the fingerprint to a csv file.
        saveCSVFile(file, args.csvDir, fingerprint)
    
    print("Done!")

#The function called if we want to search instead.
def searchFingerprints(args):
    #Load the csvs requested.
    fingerprints, names = loadCSVFiles(args.csvDir)
    fingerprint = loadtxt(args.fingerprint, delimiter=",")

    #Check if we should use the Neural dimensions or not.
    if args.fingerprinter == "Neural" and args.annoy == False:
        #Create the voyager searcher with the corresponding parameters.
        searcher = voyager(fingerprints, names, neighbours=args.numNeighbours, space=Space.Cosine)
    elif args.fingerprinter == "Traditional" and args.annoy == False:
        #Create the voyager searcher with the corresponding parameters.
        searcher = voyager(fingerprints, names, numDimensions=64, neighbours=args.numNeighbours, space=Space.Cosine)
    elif args.fingerprinter == "Neural" and args.annoy == True:
        #Create the voyager searcher with the corresponding parameters.
        searcher = annoy(fingerprints, names, neighbours=args.numNeighbours, space="angular")
    elif args.fingerprinter == "Traditional" and args.annoy == True:
        #Create the voyager searcher with the corresponding parameters.
        searcher = annoy(fingerprints, names, numDimensions=64, neighbours=args.numNeighbours, space="angular")
    #Invoke the searcher with the fingerprint.
    songs, dists = searcher.Invoke(fingerprint)

    #Loop throguh the songs and print them out.
    for i, song in enumerate(songs):
        print(song + " is " + str((1 - dists[i])*100) + " % Similar to the input song.")
    print("Done!")

#Set up argparser.
parser = ArgumentParser(prog="Hedgehog", description="Analyses and searches through tracks to find and recommend similar tracks.")

#Create a subparser so we can start adding subcommands.
subparsers = parser.add_subparsers(title="Subcommands", description="Pick between searching and fingerprinting")

#Set up fingerprint subcommand.
fingerprint = subparsers.add_parser("Fingerprint", help="Fingerprint an audio file uisng the fingerprinter of your choice.")

#Create another subparser so we can choose the type of fingerprinter. I should probably come up with a more flexible way to do this that doesn't rely on manual subcommand defintions for eacher fingerprinter.
type = fingerprint.add_subparsers(title="Fingerprinter", help="Select the fingerprinter to use.")

#Create definition for the neural fingerprinter's subcommand.
neural = type.add_parser("Neural", help="Use the Neural fingerprinter.")

#Set up the arguments for the aformentioned subcommand.
neural.add_argument("audioDir", help="Root directory of music library.", type=Path)
neural.add_argument("-c", "--csvDir", help="Directory to save the csv files to.", type=Path, default=None)
neural.add_argument("-f", "--format", help="File extension/format of the audio files to read.", type=str, default="flac")
neural.add_argument("-m", "--model", help="Path to model file.", default="./msd-musicnn-1.pb", type=Path)
neural.set_defaults(func=neuralFingerprint)

#Doing the same thing we did with the neural command with this one.
traditional = type.add_parser("Traditional", help="Use the Traditional fingerprinter.")

#Ditto.
traditional.add_argument("audioDir", help="Root directory of music library.", type=Path)
traditional.add_argument("-c", "--csvDir", help="Directory to save the csv files to.", type=Path, default=None)
traditional.add_argument("-f", "--format", help="File extension/format of the audio files to read.", type=str, default="flac")
traditional.set_defaults(func=tradFingerprint)

#Create a search subcommand. Likewise as with the fingerprinters. I just forgot that I had to convert this as well, woops.
search = subparsers.add_parser("Search", help="Search a list of output csv files to find the nearest neighbours to a song.")

#Ditto.
search.add_argument("csvDir", help="Directory to load the csv files from.", type=Path)
search.add_argument("fingerprint", help="The CSV Fingerprint for a single song.", type=Path)
search.add_argument("-k", "--numNeighbours", help="The number of neighbours to return from the query.", type=int, default=10)
search.add_argument("-f", "--fingerprinter", help="Select which fingerprinter the songs were processed with. This affects the input dimensions.", choices=("Neural", "Traditional"), default="Neural")
search.add_argument("-a", "--annoy", help="Use the alternate annoy indexer.", default=True, type=bool)
search.set_defaults(func=searchFingerprints)

#Parse args.
args = parser.parse_args()

#Call the associated function with the new arguments.
args.func(args)

