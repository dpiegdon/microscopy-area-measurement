# fdm=indent foldlevel=1 foldnestmax=2

import numpy as np
import glob
import pickle

from skimage import io, img_as_float
from skimage import color, filters
from sklearn.mixture import GaussianMixture

class train_models():
    def __init__(self, path):
        super().__init__()
        self.path = path

        model_image_globs = {}
        for m in glob.glob("{}/model_*".format(self.path)):
            model_name = m.split("model_")[1]
            print("found model '{}' in '{}'.".format(model_name, m))
            model_image_globs[model_name] = m + "/*"

        self.images = {}
        for m in model_image_globs:
            self.images[m] = io.imread_collection(model_image_globs[m])
            if len(self.images[m]) == 0:
                del self.images[m]
                print("removed {} as model has no samples.".format(m))
            else:
                print("loaded {} samples for model {}.".format(len(self.images[m]), m))

    def run(self):
        if len(self.images) == 0:
            print("no training data?")
            return -1

        segment_points10d = {}
        for m in self.images:
            segment_points10d[m] = []
            print("retrieving points for model '{}'".format(m))
            for rgb_image in self.images[m]:
                # ensure correct format and save alpha-channel
                rgb_image = img_as_float(rgb_image)
                if rgb_image.shape[2] == 4:
                    alpha = rgb_image[:,:,3]
                    rgb_image = rgb_image[:,:,0:3]
                elif rgb_image.shape[2] == 3:
                    alpha = np.ones(rgb_image.shape[0:2], dtype=float)
                else:
                    raise ValueError("Training images must have 3 or 4 channels (RGB or RGB+Alpha).")
                xyz_image = color.convert_colorspace(rgb_image, "rgb", "xyz")
                hsv_image = color.convert_colorspace(rgb_image, "rgb", "hsv")
                edges = filters.scharr(color.rgb2gray(xyz_image))
                structure_image = img_as_float(filters.median(edges, np.ones( (7,7) )))
                image11d = np.dstack((rgb_image, xyz_image, hsv_image, structure_image, alpha))
                points11d = image11d.reshape(-1,11)
                points10d = points11d[:,0:10]
                # filter non-opaque points:
                opaque_points11d = points11d[ 1. == points11d[:,10] ]
                opaque_points10d = opaque_points11d[:,0:10]
                segment_points10d[m].append(opaque_points10d)
            segment_points10d[m] = np.vstack(segment_points10d[m])
            print("  got {} points for model '{}'".format(segment_points10d[m].shape[0], m))

        models = {}
        for segment in segment_points10d:
            print("generating model '{}'".format(segment))
            models[segment] = GaussianMixture(1).fit(segment_points10d[segment])

        for m in models:
            filename = "{}/model-{}.gmmpickle".format(self.path, m)
            pickle.dump(models[m], open(filename, "wb"))
            print("model '{}' saved as {}".format(m, filename))

        return 0

