import numpy as np
from porespy.filters import snow_partitioning_parallel
from skimage.segmentation import find_boundaries
from extractPoreRegion import PoreRegionExtractor 

class UnresolvedRegionProcessor:
    def __init__(self, image, factor, pore_grayvalue, grain_grayvalue, unresolved_grayvalue):
        self.image = np.copy(image)
        self.factor = factor
        self.PORE = pore_grayvalue
        self.GRAIN = grain_grayvalue
        self.UNRESOLVED = unresolved_grayvalue
        self.poreRegions = None
        self.regions = None
        self.region_id = []
        self.neighbors_id = []
        self.neighbors_count = []
        self.neighbors_pore = []
        self.connectedRegions = []
        self.previousRegions = []
        self.cubes = []
        self.phases = []
        self.firstPoint = []

    def process_unresolved(self):
        self._prepare_image()
        self._partition_regions()
        self._analyze_regions()
        self._find_connected_regions()
        self._extract_phases()
        smallPhases = self._get_small_phases()
        return (self.regions, self.cubes, smallPhases, self.firstPoint, 
                self.connectedRegions, [self.region_id, self.neighbors_id, 
                                        self.neighbors_count, self.neighbors_pore])

    def _prepare_image(self):
        self.poreRegions = PoreRegionExtractor(self.image).extract_regions()
        grains = np.where(self.image == self.GRAIN)
        unresolvedPores = np.where(self.image == self.UNRESOLVED)
        self.image[unresolvedPores] = self.GRAIN
        self.image[grains] = self.PORE

    def _partition_regions(self):
        r_max = 4
        sigma = 0.4
        parallelization = {}
        snow = snow_partitioning_parallel(im=self.image, sigma=sigma, r_max=r_max, **parallelization)
        self.regions = np.zeros_like(snow.regions, dtype=int)
        self.regions += snow.regions + self.regions.max() * (snow.regions > 0)

    def _analyze_regions(self):
        for region in np.unique(self.regions):
            if region == 0:
                continue
            eachRegion = np.zeros(np.shape(self.regions), dtype=bool)
            eachRegion[np.where(self.regions == region)] = True
            outerBoundary = np.where(find_boundaries(eachRegion, mode='outer'))
            poreValue = self.poreRegions[outerBoundary]
            poreNeighbors = np.unique(poreValue)
            neighborsValue = self.regions[outerBoundary]
            neighbors = np.unique(neighborsValue)
            neighborsCount = [np.count_nonzero(neighborsValue == neighbor) for neighbor in neighbors]
            self.region_id.append(region)
            self.neighbors_id.append(neighbors)
            self.neighbors_count.append(neighborsCount)
            self.neighbors_pore.append(poreNeighbors)

    def _find_connected_regions(self):
        regionsCP = np.copy(np.unique(self.regions))
        regionsCP = np.delete(regionsCP, np.where(regionsCP == 0))
        counter = 0

        while len(regionsCP) > 0:
            self.previousRegions.append([regionsCP[0]])
            self.connectedRegions.append([[regionsCP[0], regionsCP[0]]])
            regionsCP = self._get_neighbors(regionsCP, regionsCP[0])
            regionsCP = np.delete(regionsCP, np.where(regionsCP == regionsCP[0]))
            counter += 1

    def _get_neighbors(self, regionsCP, region_id):
        counter = len(self.previousRegions) - 1
        for nei in self.neighbors_id[region_id - 1]:
            if nei == 0:
                continue
            if nei not in self.previousRegions[counter]:
                self.previousRegions[counter].append(nei)
                self.connectedRegions[counter].append([nei, region_id])
                regionsCP = self._get_neighbors(regionsCP, nei)
                regionsCP = np.delete(regionsCP, np.where(regionsCP == nei))
        return regionsCP

    def _extract_phases(self):
        for region in range(1, np.max(self.regions) + 1):
            phase = np.where(self.regions == region)
            firstPixel = [np.min(phase[0]), np.min(phase[1]), np.min(phase[2])]
            self.firstPoint.append(firstPixel)
            cubeWidth = np.max(phase[0]) - np.min(phase[0]) + 1
            cubeHeight = np.max(phase[1]) - np.min(phase[1]) + 1
            cubeDepth = np.max(phase[2]) - np.min(phase[2]) + 1
            self.cubes.append([cubeWidth, cubeHeight, cubeDepth])
            for i in range(len(phase)):
                phase[i][:] -= firstPixel[i]
            self.phases.append(phase)

    def _get_small_phases(self):
        smallPhases = []
        for phase in self.phases:
            smallPhasesX = []
            smallPhasesY = []
            smallPhasesZ = []
            for i in range(len(phase[0])):
                smallPhasesX.append(int(phase[0][i]))
                smallPhasesY.append(int(phase[1][i]))
                smallPhasesZ.append(int(phase[2][i]))
            smallPhases.append((np.array(smallPhasesX), np.array(smallPhasesY), np.array(smallPhasesZ)))
        return smallPhases