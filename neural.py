from argparse import ArgumentParser
from librosa import load
from librosa.feature import melspectrogram
from numpy import savetxt, random, mean, cov, moveaxis, dtypes
from pathlib import Path
from tensorflow import lite

parser = ArgumentParser(prog="Hedgehog Neural", description="Uses AI to analyse audio tracks.")

parser.add_argument('audioDir', help="Root directory of music library.", type=Path)
parser.add_argument('-c', '--csvDir', help="Directory to save the csv files to.", type=Path, default=None)
parser.add_argument("-m", "--model", help="Path to model file.", default="./Music.tflite")
parser.add_argument("-f", "--format", help="File extension/format of the audio files to read.", default="flac")

args = parser.parse_args()

interpreter = lite.Interpreter(model_path=args.model)
interpreter.resize_tensor_input(0, (2, 187, 96))
interpreter.allocate_tensors()

inputDetails = interpreter.get_input_details()[0]
outputDetails = interpreter.get_output_details()[0]

rng = random.default_rng()

for file in args.audioDir.rglob(f"*.{args.format}"):
    print(file)

    audio = load(file, mono=False)
    audioData = audio[0]

    mel = melspectrogram(
        y=audioData,
        n_mels=96,
        hop_length=512,
        win_length=1024
    )

    normal = moveaxis(rng.multivariate_normal(mean(audioData, axis=1), cov(audioData), size=(187, 96)).astype(dtypes.Float32DType), [1, 2], [-1, 0])

    interpreter.set_tensor(inputDetails['index'], normal)

    interpreter.invoke()

    outputData = interpreter.get_tensor(outputDetails['index'])

    savetxt(
        str(file.with_suffix(".csv")) if not args.csvDir else f"{args.csvDir / (file.name.rsplit('.', 1)[0])}.csv",
        outputData.reshape(-1),
        delimiter=","
    )