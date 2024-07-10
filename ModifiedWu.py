from PIL import Image
import numpy as np
import glob
import extractImagesWu
import time

class ImageProcessor:
    def __init__(self, input_path, image_format='*.tif'):
        self.input_path = input_path
        self.image_format = image_format
        self.imgs_path = self._get_image_paths()
        self.width, self.height, self.depth = self._get_image_dimensions()
        self.main_image = self._load_images()

    def _get_image_paths(self):
        imgs_path = glob.glob(self.input_path + self.image_format)
        imgs_path.sort()
        return imgs_path

    def _get_image_dimensions(self):
        I = Image.open(self.imgs_path[0])
        I = I.convert("L")
        width, height = np.shape(I)
        depth = len(self.imgs_path)
        return width, height, depth

    def _load_images(self):
        main_image = np.zeros(shape=[self.width, self.height, self.depth])
        for count, img_path in enumerate(self.imgs_path):
            I = Image.open(img_path)
            I = I.convert("L")
            main_image[:, :, count] = I
        return main_image

    def get_gray_values(self):
        gray_values = np.sort(np.unique(self.main_image))
        return gray_values

    def get_pores(self, gray_values):
        PORE = gray_values[0]
        pores = np.where(self.main_image == PORE)
        return pores

class DataLoader:
    def __init__(self, output_path):
        self.output_path = output_path
        self.regions = self._load_data('regions.npy')
        self.cubes = self._load_data('cubes.npy')
        self.phases = self._load_data('phases.npy', as_list=True)
        self.first_point = self._load_data('firstPoint.npy')
        self.connected_regions = self._load_data('connectedRegions.npy', as_list=True)
        self.neighbors = self._load_data('neighbors.npy', as_list=True)

    def _load_data(self, file_name, as_list=False):
        data = np.load(self.output_path + file_name, allow_pickle=True)
        if as_list:
            data = np.ndarray.tolist(data)
        return data

class HighResImageProcessor:
    def __init__(self, low_res_processor, data_loader, hr_image_path, image_format, output_path):
        self.low_res_processor = low_res_processor
        self.data_loader = data_loader
        self.hr_image_path = hr_image_path
        self.image_format = image_format
        self.output_path = output_path
        self.high_res_images = None
        self.cluster_cubes = None
        self.cluster_start_points = None
        self.cluster_phases = None

    def extract_images(self):
        self.high_res_images, self.cluster_cubes, self.cluster_start_points, self.cluster_phases = extractImagesWu.getImages(
            self.low_res_processor.main_image,
            self.hr_image_path,
            self.output_path,
            self.image_format,
            self.data_loader.cubes,
            self.data_loader.phases,
            self.data_loader.first_point,
            self.data_loader.connected_regions,
            self.data_loader.neighbors
        )
        np.save(self.output_path + 'highresImagesWu.npy', self.high_res_images)

    def merge_images(self):
        main_image = self.low_res_processor.main_image
        for i in range(len(self.cluster_cubes)):
            x, y, z = self.cluster_start_points[i]
            dx, dy, dz = self.cluster_cubes[i]
            phase = self.cluster_phases[i]
            main_image[x:x+dx, y:y+dy, z:z+dz][phase] = self.high_res_images[i][phase]

    def save_images(self, prefix="Wu"):
        main_image = self.low_res_processor.main_image
        for i in range(np.shape(main_image)[2]):
            save_image = Image.fromarray(main_image[:, :, i])
            save_image = save_image.convert("L")
            save_image.save(f'{self.output_path}/{prefix}{i:04d}.png')

def main():
    start_time = time.time()

    output_path = "./path_to_output_folder/"
    low_res_image_path = './path_to_Resampled_LR_image_folder/'
    hr_image_path = './path_to_HR_image_folder/'
    image_format = '*.tif'

    low_res_processor = ImageProcessor(low_res_image_path, image_format)
    data_loader = DataLoader(output_path)
    hr_processor = HighResImageProcessor(low_res_processor, data_loader, hr_image_path, image_format, output_path)

    hr_processor.extract_images()
    hr_processor.merge_images()
    hr_processor.save_images()

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
