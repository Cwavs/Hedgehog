# Hedgehog
## Very WIP Code is not clean and probably needs alot of improvement.

A (bad) attempt at reverse engineering/re-creating [Plex's Sonic Analysis](https://www.plex.tv/en-gb/blog/super-sonic-get-closer-to-your-music-in-plexamp/) features in an open way.

## Requirements

Only tested on Python 3.12.8

Hedgehog utilises a couple of python modules which can be obtained with pip:

```bash
pip install -r requirements.txt
```

Tensorflow is not needed for traditional fingerprinter, as that works without the neural model. I it made for testing however, so this is not the focus of the project.

The neural fingerprinter currently relies on Plex's Sonic Analysis model as well. We can't provide that so you'll have to find your own copy somewhere.

## Usage

The project is currently split into two scripts which need to be run individually, the fingerprinter and the searcher.

The fingerprinter analyses the music and saves the data to a series of CSV files in a directory of your choice (placed next to the media by default).
The below command would fingerprint all mp3 files in /home/cwavs/music with the neural fingerprinter using Music.tflite as the model. It would save the csv files /home/cwavs/csvdir

```bash
python Hedgehog Fingerprint Neural /home/cwavs/music -c /home/cwavs/csvdir -m Music.tflite -f mp3
```

The searcher then loads the CSV data and uses voyager to index them. It can then use any of the CSV files to find the 10 closest songs to them, each representing a song. You can select the number of neighbours with -k
The below command would print 100 neighbours to song.csv

```bash
python Hedgehog Search /home/cwavs/csvdir /home/cwavs/csvdir/song.csv -k 100 -f Neural
```

## TODO

- [X] Allow users to select the amount of neighbours chosen.
- [X] Investigate possible improvements to the neural fingerprinter with better pre-processing.
- [X] Generally clean up the code, and maybe consider converting it into an actual module or library rather than scripts.

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
