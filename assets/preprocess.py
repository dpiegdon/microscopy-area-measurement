# fdm=indent foldlevel=1 foldnestmax=2

import numpy as np
import multiprocessing
import glob

from skimage import io, img_as_ubyte
from skimage import restoration

from . import filenames
from . import clear

class preprocess():
    def __init__(self, path):
        super().__init__()
        self.path = path

        clear.clear_preprocessed_images(self.path)

        self.all_images = glob.glob(filenames.raw_image_glob(self.path))
        print("Selected {} images to preprocess.".format(len(self.all_images)))

    def preprocess_image(self, imagefilename):
        rgb_image = img_as_ubyte(io.imread(imagefilename))
        rgb_image = restoration.denoise_bilateral(rgb_image, multichannel=True)
        if rgb_image.shape[2] == 3:
            alpha_channel = np.ones(rgb_image.shape[:-1])
            rgba_image = np.dstack((rgb_image, alpha_channel))
        else:
            rgba_image = rgb_image

        io.imsave(filenames.preprocessed_file(imagefilename), rgba_image)

    def run(self):
        threads = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(threads, maxtasksperchild=1) as p:
            p.map(self.preprocess_image, self.all_images, chunksize=1)

