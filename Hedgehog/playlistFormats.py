from pathlib import Path

#Just creating a standard class here to dervive others from. Only implementing M3U for now, but would like to implement others like JSPF later.
class _playlistFormat():

    #Set creation parameters.
    def __init__(self, savePath: Path, songList: list, fileExtension: str) -> None:
        #Store values.
        self.songList = songList
        self.savePath = savePath
        self.fileExtension = fileExtension
    
    #Created a generic Save function to override. This doesn't do anything here as it is format specific.
    def Save(self) -> bool:
        return False

class M3U(_playlistFormat):

    def __init__(self, savePath: Path, songList: list, relativePath: Path, fileExtension: str = "m3u", absolutePaths: bool = True) -> None:
        super().__init__(savePath, songList, fileExtension)
        self.relativePath = relativePath
        self.absolutePaths = absolutePaths
    
    def Save(self) -> bool:
        with open(self.savePath.with_suffix(self.fileExtension), "w") as file:
            if self.absolutePaths == True:
                print("Using Absolute Paths.")
                for song in self.songList:
                    file.write(str(song.resolve()) + "\n")
            else:
                print("Using Relative Paths.")
                for song in self.songList:
                    file.write(str(song.relative_to(str(self.relativePath))) + "\n")
            file.close()