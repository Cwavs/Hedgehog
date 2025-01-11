from argparse import ArgumentParser
from numpy import loadtxt
from voyager import Index, Space
from pathlib import Path

#Setup argparser.
parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

#Set up arugments to be parsed.
parser.add_argument('csvDir', help="Directory to load the csv files from.", type=Path)
parser.add_argument('fingerprint', help="The CSV Fingerprint for a single song.", type=Path)
parser.add_argument('-k', '--numNeighbours', help="The number of neighbours to return from the query.", type=int, default=10)
parser.add_argument('-f', '--fingerprinter', help="Select which fingerprinter the songs were processed with. This affects the input dimensions. 0 is the neural fingerprinter and 1 is the traditional fingerprinter.", type=int, default=0)

#Parse args.
args = parser.parse_args()

#Decide what to do based on which fingerprinter the user has chosen.
if args.fingerprinter == 0:
    #If they picked the neural fingerprinter, then the number of input dimesnions is 50.
    index = Index(Space.Euclidean, num_dimensions=50, ef_construction=5000)
elif args.fingerprinter == 1:
    #If they picked the traditional fingerprinter, then the number of inputer dimensions is 64.
    index = Index(Space.Euclidean, num_dimensions=64, ef_construction=5000)
else:
    #If they picked something else, they are lying and we need to tell them to pick a real fingerprinter and exit.
    print("Error, invalid fingerprinter.")
    exit()

#Instantiate a list to store the names of the songs/audio files in.
names = list()

#Loop through the directory provided for .csv files.
for file in args.csvDir.rglob("*.csv"):
    #Load the csv file as a numpy array.
    single = loadtxt(file, delimiter=",")
    #Append the file name to the names list.
    names.append(file.name)
    #Add the item to the index with. We set the id to the corresponding index in the names list. This way we can easily recall which song is being referenced.
    index.add_item(single, id=len(names)-1)

#Load in the requested csv file to search for neighbours too.
single = loadtxt(args.fingerprint, delimiter=",")

#Query the index for the nearest neighbours. I need to look more into how much the ef affects the results. I am not sure how important it is. 5000 seems fine for now (the same applies to ef_construction when building the array).
songs, dists = index.query(single, k=args.numNeighbours, query_ef=5000)

#Loop through the query results and print out the neighbours names and how far they are from the input.
for i, song in enumerate(songs):
    print(names[song] + " is " + str(dists[i]) + " away from the input.")