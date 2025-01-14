from voyager import Index, Space
from numpy import ndarray

#Create a base searcher class.
class _searcher():
    
    #Set creation parameters.
    def __init__(self, fingerprints: list, names: list, neighbours: int = 10) -> None:
        #Store values.
        self.fingerprints = fingerprints
        self.names = names
        self.neighbours = neighbours

    #Define a get neighbours function to override later.
    def getNeighbours(self) -> tuple:
        #Unlike the fingerprinter and preprocessor, I couldn't come up with a generic result for this (like just returnign the raw audio). So instead I thought it would be funny to return 10 random values.
        from random import randrange

        songs = list()
        dist = list()

        for i in range(0, self.neighbours):
            songs.append(self.names[randrange(0, len(self.names))])
            dist.append(i)
        
        return (songs, dist)
    
    #Define an invocation function to be overridden later.
    def Invoke(self) -> tuple:
        #Here I'm just calling get neighbours and returning the results.
        songs, dists = self.getNeighbours()
        
        return (songs, dists)

#Create voyager class as an override of the base searcher class.
class voyager(_searcher):

    #Set creation parameters, adding additional voyager specific parameters.
    def __init__(self, fingerprints: list, names: list, neighbours: int = 10, space: Space = Space.Euclidean, numDimensions: int = 50) -> None:
        #Call the parent's init to store it's values itself.
        super().__init__(fingerprints, names, neighbours)

        #Store the rest of the values.
        self.space = space
        self.numDimensions = numDimensions
        
    #Define a function to conver Voyager IDs into song names.
    def _IDsToNames(self, songIDs: list) -> list:
        songs = list()

        #Simply loop through each songID and append the corresponding name.
        for songID in songIDs:
            songs.append(self.names[songID])
        
        return songs

    #Define a function to build a voyager index.
    def buildIndex(self) -> Index:
        from numpy import stack
        
        #Create an index containing the space and number of dimensions defined in the class initialisation.
        index = Index(self.space, num_dimensions=self.numDimensions)
        #Add fingerprints from list provided during initalisation, and generate a list of IDs to use.
        index.add_items(self.fingerprints, ids=range(0, len(self.names)))

        return index

    def getNeighbours(self, song: ndarray, index: Index) -> tuple:
        #Query the index for the defined number of neighbours.
        results = index.query(song, k=self.neighbours)

        return results

    def Invoke(self, song: ndarray) -> tuple:
        #Build the index.
        index = self.buildIndex()
        #Query the index we just built.
        songs, dists = self.getNeighbours(song, index)
        #Covert the song IDs into names.
        songs = self._IDsToNames(songs)

        return (songs, dists)
