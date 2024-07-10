import numpy as np
from porespy.filters import snow_partitioning_parallel
from porespy.tools import make_contiguous
from porespy.networks import add_boundary_regions

class PoreRegionExtractor:
    def __init__(self, image, boundary_width=3, sigma=0.4, r_max=4, parallelization=None):
        self.image = np.copy(image)
        self.boundary_width = boundary_width
        self.sigma = sigma
        self.r_max = r_max
        self.parallelization = parallelization if parallelization is not None else {}
        self.regions = None

    def extract_regions(self):
        self._prepare_image()
        vals = self._get_unique_values()
        self._partition_and_aggregate_regions(vals)
        self._add_boundary_regions()
        return make_contiguous(self.regions)

    def _prepare_image(self):
        self.image = np.array(self.image, dtype=bool)
        self.image = ~self.image
        self.image = self.image.astype(int)

    def _get_unique_values(self):
        vals = np.unique(self.image)
        return vals[vals > 0]

    def _partition_and_aggregate_regions(self, vals):
        for i in vals:
            phase = self.image == i
            snow = snow_partitioning_parallel(im=phase, sigma=self.sigma, r_max=self.r_max, **self.parallelization)
            if self.regions is None:
                self.regions = np.zeros_like(snow.regions, dtype=int)
            self.regions += snow.regions + self.regions.max() * (snow.regions > 0)

        if self.image.shape != self.regions.shape:
            for ax in range(self.image.ndim):
                self.image = np.swapaxes(self.image, 0, ax)
                self.image = self.image[:self.regions.shape[ax], ...]
                self.image = np.swapaxes(self.image, 0, ax)

    def _add_boundary_regions(self):
        boundary_width = self._parse_pad_width(self.boundary_width, self.image.shape)
        if np.any(boundary_width):
            self.regions = add_boundary_regions(self.regions, pad_width=boundary_width)
            self.image = np.pad(self.image, pad_width=boundary_width, mode='edge')

    def _parse_pad_width(self, pad_width, shape):
        ndim = len(shape)
        pad_width = np.atleast_1d(np.array(pad_width, dtype=object))

        if np.size(pad_width) == 1:
            pad_width = np.tile(pad_width.item(), ndim).astype(object)
        if len(pad_width) != ndim:
            raise Exception(f"pad_width must be scalar or {ndim}-element list")

        tmp = []
        for elem in pad_width:
            if np.size(elem) == 1:
                tmp.append(np.tile(np.array(elem).item(), 2))
            elif np.size(elem) == 2 and np.ndim(elem) == 1:
                tmp.append(elem)
            else:
                raise Exception("pad_width components can't have 2+ elements")

        return np.array(tmp)
