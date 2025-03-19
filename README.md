# Hedgehog
## Not as very WIP Code is not clean and probably needs alot of improvement.

A (bad) python command line program/library built around Essentia/MusiCNN's music auto-tagging model. You probably know this better as the model that powers [Plex's Sonic Analysis](https://www.plex.tv/en-gb/blog/super-sonic-get-closer-to-your-music-in-plexamp/) feature.

## Requirements

Due to an issue with Essedtia's Python library, this program is currently incompatible with Windows. Windows users will need to  use WSL for now. In the future I plan to also provide a docker image.

Only tested on Python 3.11.11

Hedgehog utilises a couple of python modules which can be obtained with pip:

```bash
pip install -r requirements.txt
```

In order to use the neural fingerprinter, you will also need the corresponding model. You can obtain a copy from Essentia here: [https://essentia.upf.edu/models.html#msd-musicnn](https://essentia.upf.edu/models.html#msd-musicnn)

## Usage

For most users currently, the project works in two steps. Fingerprinting and Searching.

Fingerprinting is the process of loading each file and passing it through the model, and then doing some light post-processing on the results.

This fingerprint of the songs (or embeddings which is probably the more correct term) represent the various characterisitics of the music. You can find a list of these tags on this essentia.js demo. We save this to CSV files next to the song, or alternatively in a directory of your choice.

The below command would fingerprint all mp3 files in /home/cwavs/music using Music.tflite as the model. It would save the CSV files /home/cwavs/csvdir

```bash
python Hedgehog Fingerprint Neural /home/cwavs/music -c /home/cwavs/csvdir -m msd-musicnn-1.pb -f mp3
```

Searching then is the process of loading the CSVs up and constructing an index so we can traverse them and find the closest songs to a given song. You can think of this like generating a playlist of similar sounding songs when given one, or recommending songs based on one your friend likes.

We currently offer two solutions for this. Both are libraries developed by spotify. We use Annoy (which is what Plex uses) and Voyager (the successor to Annoy). We use both with cosine/angular space respectivley. Empiraclly Annoy seems to give slightly better results than Voyager, with the distances often seeming more accurate to me, as such Annoy is currently the default. This might be something we can fix by adjusting Voyager's parameters though.

We also currently only support finding the nearest neighbours to one song. I would very much like to add more advanced features though.

The below command would load all CSVs in the directory /home/cwavs/csvdir and construct an annoy Index for the Neural fingerprinter. It then loads the CSV "song.csv" and finds the 100 closest songs too it.

```bash
python Hedgehog Neighbours /home/cwavs/csvdir /home/cwavs/csvdir/song.csv -k 100 -f Neural
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is lisenced with the LGPL 3: https://choosealicense.com/licenses/lgpl-3.0/

## Special Thanks

The friends who generously donated their music libraries to testing (you know who you are).
The asshole who initially tried to reverse engineer it and shilled Sonic Analysis so much I got bored of waiting for someone else to do this (you also know who you are).
The talented indivduals behind [Musly](https://www.musly.org/) whose work helped kickstart my understanding of music similarity and fingerprinting.
Essentia, despite the headaches your python library gave me, you guys do some pretty cool work, and none of this would be possible without you.
