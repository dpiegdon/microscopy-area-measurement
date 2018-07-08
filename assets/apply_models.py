# fdm=indent foldlevel=1 foldnestmax=2

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import collections
import glob
import pickle
import pathlib

from skimage import io, img_as_float
from skimage import color, filters, morphology
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob

class apply_models():
    def __init__(self, path):
        super().__init__()
        self.path = path

        self.models = collections.OrderedDict()
        for m in glob.glob("{}/model-*.gmmpickle".format(self.path)):
            model_name = m.split("model-")[1].replace(".gmmpickle", "")
            print("loading model '{}' from '{}'.".format(model_name, m))
            self.models[model_name] = pickle.load( open(m, "rb") )

        self.all_images = [f for f in pathlib.Path(self.path).glob("*_preprocessed.png")]
        print("selected {} images from '{}' to segment.".format(
                                len(self.all_images),
                                self.path
                            ))

    def apply_model(self, filename):
        def multivariate_gaussian_prediction(gmm, X):
            return np.exp(
                        _estimate_log_gaussian_prob(
                                X,
                                gmm.means_,
                                gmm.precisions_cholesky_,
                                gmm.covariance_type
                            )
                    )

        min_object_size=75

        rgb_image = io.imread(filename)
        shape = (rgb_image.shape[0], rgb_image.shape[1])
        if rgb_image.shape[2] == 4:
            # remove alpha channel
            rgb_image = img_as_float(rgb_image[:,:,0:-1])
        elif rgb_image.shape[2] == 3:
            rgb_image = img_as_float(rgb_image)
        else:
            raise ValueError("Images must have 3 or 4 channels (RGB or RGB+Alpha; Alpha is ignored).")
        xyz_image = color.convert_colorspace(rgb_image, "rgb", "xyz")
        hsv_image = color.convert_colorspace(rgb_image, "rgb", "hsv")
        structure_image = img_as_float(
                                filters.median(
                                        filters.scharr(
                                                color.rgb2gray(xyz_image)
                                        ),
                                        np.ones( (7,7) )
                                )
                          )
        image10d = np.dstack((rgb_image, xyz_image, hsv_image, structure_image))
        points10d = image10d.reshape(-1,10)
        result = {}
        model = {}
        for i, m in enumerate(self.models, start=1):
            model[m] = {}
            model[m]["id"] = i
        # calculate model likelihoods:
        for m in self.models:
            model[m]["likelihood"] = np.reshape(
                                            multivariate_gaussian_prediction(
                                                self.models[m],
                                                points10d
                                            ),
                                            shape
                                        )
        # calculate model masks:
        modelstack = np.zeros(shape)
        for m in self.models:
            mask = np.ones(shape, dtype=bool)
            for other in self.models:
                if m is not other:
                    not_member = model[m]["likelihood"] < model[other]["likelihood"]
                    mask[not_member] = False
            modelstack[ mask ] = model[m]["id"]
        for m in self.models:
            # remove small specks from model masks:
            mask = modelstack == model[m]["id"]
            mask_fixes = np.logical_not(
                             morphology.remove_small_objects(
                                 np.logical_not(mask), min_size=min_object_size ) )
            # calculate area of each model
            pixels = np.sum(mask_fixes)
            percent = (100. * pixels / (mask.shape[0]*mask.shape[1]))
            result["filename"] = filename
            result["segment_"+m+"_area_pixel"]   = pixels
            result["segment_"+m+"_area_percent"] = percent
            # apply mask fixes:
            mask_fixes[mask] = False
            modelstack[ mask_fixes ] = model[m]["id"]
        # create overview plot for manual verification
        f, axes = plt.subplots(1+len(self.models), 2, figsize=(20,8*(1+len(self.models))))
        f.suptitle(filename, fontsize=20)
        (axis_org, axis_segments) = axes[0]
        axis_org.axis('off')
        axis_org.imshow(rgb_image)
        axis_org.set_title("original image", fontsize=16)
        axis_segments.axis('off')
        axis_segments.imshow(modelstack)
        axis_segments.set_title("all segments", fontsize=16)
        black_pixel = np.array([0,0,0])
        for (axis_mask, axis_maskedimage), m in zip(axes[1:], model):
            mask = modelstack == model[m]["id"]
            axis_mask.axis('off')
            axis_mask.imshow(mask)
            axis_mask.set_title("mask '{}': {} pixel".format(
                                        m, result["segment_"+m+"_area_pixel"]), fontsize=16)
            maskedimage = np.array(rgb_image)
            maskedimage[np.logical_not(mask)] = black_pixel
            axis_maskedimage.axis('off')
            axis_maskedimage.imshow(maskedimage)
            axis_maskedimage.set_title("masked image '{}': {:2.2f}%".format(
                                        m, result["segment_"+m+"_area_percent"]), fontsize=16)
        plt.axis('off')
        figure_file = str(filename).replace("_preprocessed.png", "_analysis.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(figure_file, facecolor="grey", edgecolor="black")
        return result

    def run(self):
        #threads = multiprocessing.cpu_count() - 1
        threads = 3
        with multiprocessing.Pool(threads, maxtasksperchild=1) as p:
            results = p.map(self.apply_model, self.all_images, chunksize=1)

        csv = "filename;{}\n".format(";".join(("{} area [pixels];{} area [%]".format(m,m) for m in self.models)))
        for result in results:
            csv += "{}".format(result["filename"])
            for m in self.models:
                csv += ";{};{:2.4f}".format(result["segment_"+m+"_area_pixel"],
                                     result["segment_"+m+"_area_percent"])
            csv += "\n"
        with open("{}/analysis-summary.csv".format(self.path), "w") as csv_file:
            csv_file.write(csv)

        return 0

