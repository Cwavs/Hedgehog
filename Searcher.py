from argparse import ArgumentParser
from numpy import loadtxt
from voyager import Index, Space
from pathlib import Path

parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

parser.add_argument('csvDir', help="Directory to load the csv files from.", type=Path)
parser.add_argument('fingerprint', help="The CSV Fingerprint for a single song.", type=Path)
parser.add_argument('-k', '--numNeighbours', help="The number of neighbours to return from the query.", type=int, default=10)
parser.add_argument('-f', '--fingerprinter', help="Select which fingerprinter the songs were processed with. This affects the input dimensions. 0 is the neural fingerprinter and 1 is the traditional fingerprinter.", type=int, default=0)

args = parser.parse_args()

if args.fingerprinter == 0:
    index = Index(Space.Euclidean, num_dimensions=50, ef_construction=5000)
elif args.fingerprinter == 1:
    index = Index(Space.Euclidean, num_dimensions=64, ef_construction=5000)
else:
    print("Error, invalid fingerprinter.")
    exit()

names = list()

for file in args.csvDir.rglob("*.csv"): 
    single = loadtxt(file, delimiter=",")
    names.append(file.name)
    index.add_item(single, id=len(names)-1)

single = loadtxt(args.fingerprint, delimiter=",")

songs, dists = index.query(single, k=args.numNeighbours, query_ef=5000)

for i, song in enumerate(songs):
    print(names[song] + " is " + str(dists[i]) + " away from the input.")