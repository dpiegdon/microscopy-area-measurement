# fdm=indent foldlevel=1 foldnestmax=2

import pathlib
import numpy as np
import multiprocessing

from skimage import io, img_as_ubyte
from skimage import restoration

class preprocess():
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.all_images = [f for f in pathlib.Path(self.path).glob("*.tif")]
        print("selected {} images from '{}' to preprocess.".format(
                                len(self.all_images),
                                self.path
                            ))

    def preprocess_image(self, filename):
        filename = str(filename)
        newfilename = filename.replace(".tif", "_preprocessed.png")
        rgb_image = restoration.denoise_bilateral(
                        img_as_ubyte(io.imread(filename)),
                        multichannel=True
                    )
        if rgb_image.shape[2] == 3:
            alpha_channel = np.ones(rgb_image.shape[:-1])
            rgba_image = np.dstack((rgb_image, alpha_channel))
        else:
            rgba_image = rgb_image
        io.imsave(newfilename, rgba_image)
        return None

    def run(self):
        threads = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(threads, maxtasksperchild=1) as p:
            p.map(self.preprocess_image, self.all_images, chunksize=1)

        return 0

