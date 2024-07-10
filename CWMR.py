from PIL import Image
import numpy as np
import glob
import time
from computedUnresolved import UnresolvedRegionProcessor
import extractTemplatesCWMR

class ImageProcessor:
    def __init__(self, factor, output_path, lr_image_path, hr_image_path, image_format):
        self.factor = factor
        self.output_path = output_path
        self.lr_image_path = lr_image_path
        self.hr_image_path = hr_image_path
        self.image_format = image_format
        self.mainImage = None
        self.gray_values = None
        self.UNRESOLVED = None
        self.GRAIN = None
        self.PORE = None
        self.regions = None
        self.cubes = None
        self.phases = None
        self.firstPoint = None
        self.connectedRegions = None
        self.neighbors = None
        self.highResImages = None

    def load_images(self, path):
        imgs_path = glob.glob(path + self.image_format)
        imgs_path.sort()
        I = Image.open(imgs_path[0])
        I = I.convert("L")
        width, height = np.shape(I)
        depth = len(imgs_path)
        image_stack = np.zeros(shape=[width, height, depth])
        for count, img_path in enumerate(imgs_path):
            I = Image.open(img_path)
            I = I.convert("L")
            image_stack[:, :, count] = I
        return image_stack

    def save_images(self, image_stack, prefix="M"):
        for i in range(np.shape(image_stack)[2]):
            save_image = Image.fromarray(image_stack[:, :, i])
            save_image = save_image.convert("L")
            filename = f'{self.output_path}/{prefix}{i:04d}.tif'
            save_image.save(filename)

    def process_images(self):
        self.mainImage = self.load_images(self.lr_image_path)
        self.gray_values = np.sort(np.unique(self.mainImage))
        self.UNRESOLVED = self.gray_values[1]
        self.GRAIN = self.gray_values[2]
        self.PORE = self.gray_values[0]

        grains = np.where(self.mainImage == self.GRAIN)
        self.regions, self.cubes, self.phases, self.firstPoint, self.connectedRegions, self.neighbors = UnresolvedRegionProcessor(
            self.mainImage, self.factor, self.PORE, self.GRAIN, self.UNRESOLVED
        ).process_unresolved()

        self.mainImage[grains] = self.GRAIN

        self.save_intermediate_data()

        self.highResImages = extractTemplatesCWMR.getImages(
            self.hr_image_path, self.output_path, self.image_format, self.cubes, self.phases, self.firstPoint, self.connectedRegions, self.neighbors
        )

        np.save(self.output_path + 'highresImages.npy', self.highResImages)

        self.replace_regions_with_highres()

        self.save_images(self.mainImage)

    def save_intermediate_data(self):
        np.save(self.output_path + 'regions.npy', self.regions)
        np.save(self.output_path + 'cubes.npy', self.cubes)
        np.save(self.output_path + 'phases.npy', self.phases)
        np.save(self.output_path + 'firstPoint.npy', self.firstPoint)
        np.save(self.output_path + 'connectedRegions.npy', self.connectedRegions)
        np.save(self.output_path + 'neighbors.npy', self.neighbors)

    def replace_regions_with_highres(self):
        for region in np.unique(self.regions):
            if region == 0:
                continue
            self.mainImage[
                self.firstPoint[region - 1][0]:self.firstPoint[region - 1][0] + self.cubes[region - 1][0],
                self.firstPoint[region - 1][1]:self.firstPoint[region - 1][1] + self.cubes[region - 1][1],
                self.firstPoint[region - 1][2]:self.firstPoint[region - 1][2] + self.cubes[region - 1][2]
            ][self.phases[region - 1]] = self.highResImages[region - 1][self.phases[region - 1]]

if __name__ == "__main__":
    start_time = time.time()
    
    factor = 2  # Resolution difference between LR and HR images
    output_path = "./path_to_output_folder/"
    lr_image_path = './path_to_Resampled_LR_image_folder/'
    hr_image_path = './path_to_HR_image_folder/'
    image_format = '*.tif'
    
    processor = ImageProcessor(factor, output_path, lr_image_path, hr_image_path, image_format)
    processor.process_images()
    
    print("--- %s seconds ---" % (time.time() - start_time))
