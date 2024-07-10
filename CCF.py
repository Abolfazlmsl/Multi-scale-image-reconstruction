import numpy as np
import random

class CCFCalculator:
    def __init__(self, image, cropImage, templateImage, 
                 xFirstPoint, xSecondPoint, 
                 yFirstPoint, ySecondPoint, 
                 zFirstPoint, zSecondPoint):
        self.image = image
        self.cropImage = cropImage
        self.templateImage = templateImage
        self.xFirstPoint = xFirstPoint
        self.xSecondPoint = xSecondPoint
        self.yFirstPoint = yFirstPoint
        self.ySecondPoint = ySecondPoint
        self.zFirstPoint = zFirstPoint
        self.zSecondPoint = zSecondPoint
        self.ccfArray = None

    def compute_ccf(self):
        self._initialize_ccf_array()
        self._compute_ccf_values()
        return self._get_maximum_ccf_indices()

    def _initialize_ccf_array(self):
        self.ccfArray = np.zeros((np.abs(self.xSecondPoint - self.xFirstPoint),
                                  np.abs(self.ySecondPoint - self.yFirstPoint),
                                  np.abs(self.zSecondPoint - self.zFirstPoint)))

    def _compute_ccf_values(self):
        width, height, depth = np.shape(self.image)
        Lx, Ly, Lz = np.shape(self.cropImage)
        lx, ly, lz = np.shape(self.templateImage)

        for i in range(self.xFirstPoint, self.xSecondPoint):
            for j in range(self.yFirstPoint, self.ySecondPoint):
                for k in range(self.zFirstPoint, self.zSecondPoint):
                    ccfAmount = np.multiply(
                        self.image[i:i+lx, j:j+ly, k:k+lz],
                        self.cropImage[Lx-lx:lx+Lx-lx, Ly-ly:ly+Ly-ly, Lz-lz:lz+Lz-lz]
                    )
                    self.ccfArray[i - self.xFirstPoint, j - self.yFirstPoint, k - self.zFirstPoint] = np.sum(ccfAmount)

    def _get_maximum_ccf_indices(self):
        sort_arr = np.sort(self.ccfArray.ravel())
        ccfMax = sort_arr[int(len(sort_arr)/2) - 1]
        indices = np.where(self.ccfArray == ccfMax)
        index = random.randint(0, len(indices[0]) - 1)
        return [indices[0][index], indices[1][index], indices[2][index]]
