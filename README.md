# Hedgehog

A (bad) attempt at reverse engineering/re-creating Plex's Sonic Analysis features in an open way.

## Requirements

Hedgehog utilises a couple of python modules which can be obtained with pip:

```bash
pip install librosa numpy tensorflow voyager
```

Tensorflow is technically optional as there is a more traditional fingerprinter that works without the neural model, however this is not the focus of the project.

The neural fingerprinter currently relies on Plex's Sonic Analysis model as well. We can't provide that so you'll have to find your own copy somewhere.

## Usage

The project is currently split into two scripts which need to be run individually, the fingerprinter and the searcher.

The fingerprinter analyses the music and saves the data to a series of CSV files in a directory of your choice (placed next to the media by default).

```bash
python neuralFingerprinter.py /home/cwavs/music -c /home/cwavs/csvdir -m Music.tflite -f mp3
```

The searcher then loads the CSV data and uses voyager to index them. It can then use any of the CSV files to find the 10 closest songs to them, each representing a song. It's currently fixed at 10 songs.

```bash
python searcher.py /home/cwavs/csvdir /home/cwavs/csvdir/song.csv
```

## TODO

- [ ] Allow users to select the amount of neighbours chosen.
- [ ] Investigate possible improvements to the neural fingerprinter with better pre-processing.
- [ ] Generally clean up the code, and maybe consider converting it into an actual module rather than scripts.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

TBD

## Special Thanks

The friends who generously donated their music libraries to testing (you know who you are).
The asshole who initially tried to reverse engineer it and shilled Sonic Analysis so much I got bored of waiting for someone else to do this (you also know who you are).
The talented indivduals behind [Musly](https://www.musly.org/) whose work helped kickstart my understanding of music similarity and fingerprinting.
