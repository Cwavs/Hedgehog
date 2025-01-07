from argparse import ArgumentParser
from librosa import load
from librosa.feature import mfcc
from numpy import savetxt, random, mean, cov
from pathlib import Path

parser = ArgumentParser(prog="Hedgehog", description="Fingerprints audio tracks.")

parser.add_argument('audioDir', help="Root directory of music library.", type=Path)
parser.add_argument('-c', '--csvDir', help="Directory to save the csv files to.", type=Path, default=None)
parser.add_argument("-f", "--format", help="File extension/format of the audio files to read.",type=str, default="flac")

args = parser.parse_args()

rng = random.default_rng()

for file in args.audioDir.rglob(f"*.{args.format}"):
    print(file)

    audio = load(file)
    audioData = audio[0]

    melfcc = mfcc(
        y=audioData,
        n_mfcc=4,
        hop_length=512,
        win_length=1024
    )

    normal = rng.multivariate_normal(mean(melfcc, axis=1), cov(melfcc), size=(4, 4))

    savetxt(
        str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
        normal.reshape((64)),
        delimiter=","
    )