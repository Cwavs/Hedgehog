from argparse import ArgumentParser
import numpy as np
from voyager import Index, Space
from pathlib import Path

parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

parser.add_argument('csvDir', help="Directory to load the csv files from.", type=Path)
parser.add_argument('fingerprint', help="The CSV Fingerprint for a single song.", type=Path)
parser.add_argument('-k', '--numNeighbours', help="The number of neighbours to return from the query.", type=int, default=10)

args = parser.parse_args()

index = Index(Space.Euclidean, num_dimensions=50, ef_construction=5000)
names = list()

for file in args.csvDir.rglob("*.csv"): 
    single = np.loadtxt(file, delimiter=",")
    names.append(file.name)
    index.add_item(single, id=len(names)-1)

single = np.loadtxt(args.fingerprint, delimiter=",")

songs, dists = index.query(single, k=args.numNeighbours, query_ef=5000)

for i, song in enumerate(songs):
    print(names[song] + " is " + str(dists[i]) + " away from the input.")