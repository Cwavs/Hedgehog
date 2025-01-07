from argparse import ArgumentParser
import numpy as np
from voyager import Index, Space
from pathlib import Path

parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

parser.add_argument('csvDir', help="Directory to load the csv files from.", type=Path)
parser.add_argument('fingerprint', help="The CSV Fingerprint for a single song.", type=Path)

args = parser.parse_args()

index = Index(Space.Euclidean, num_dimensions=100)
name = list()

for file in args.csvDir.rglob("*.csv"): 
    single = np.loadtxt(file, delimiter=",")
    name.append(file.name)
    index.add_item(single, id=len(name)-1)

single = np.loadtxt(args.fingerprint, delimiter=",")

songs = index.query(single, k=10)

for song in songs[0]:
    print(name[song])