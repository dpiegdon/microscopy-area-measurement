# fdm=indent foldlevel=1 foldnestmax=2

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import collections
import glob
import pickle
import os

from skimage import io, img_as_float
from skimage import color, filters, morphology
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob

from . import filenames


def _multivariate_gaussian_prediction(gmm, X):
    return np.exp(
                _estimate_log_gaussian_prob(
                        X,
                        gmm.means_,
                        gmm.precisions_cholesky_,
                        gmm.covariance_type
                    )
            )

def _get_preprocessed_image(image_file):
    rgb_image = io.imread(image_file)
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

    return (shape, rgb_image, points10d)

def _calc_model_likelihoods(shape, points10d, models):
    likelihoods = {}

    for i, m in enumerate(models, start=1):
        prediction = _multivariate_gaussian_prediction(models[m], points10d)
        likelihoods[m] = np.reshape(prediction, shape)

    return likelihoods

_color_index_palette = [
    # segment 0 (black) is reserved and should never be used
    np.array([000,000,000], dtype=np.uint8), # black
    np.array([255,000,000], dtype=np.uint8), # red
    np.array([000,255,000], dtype=np.uint8), # green
    np.array([000,000,255], dtype=np.uint8), # blue
    np.array([000,255,255], dtype=np.uint8), # cyan
    np.array([255,000,255], dtype=np.uint8), # magenta
    np.array([255,255,000], dtype=np.uint8), # yellow
    np.array([255,255,255], dtype=np.uint8), # white
]

def _image_indexed2color(image_xy):
    # returns image in uint8 format
    result = np.zeros([image_xy.shape[0], image_xy.shape[1], 3], dtype=np.uint8)
    done = np.zeros([image_xy.shape[0], image_xy.shape[1]], dtype=np.bool)
    for i, color in enumerate(_color_index_palette, start=0):
        mask = (image_xy == i)
        result[mask] = color
        done[mask] = True;

    pixels = image_xy.shape[0] * image_xy.shape[1]
    pixels_done = np.sum(done)
    if(pixels_done != pixels):
        raise ValueError("Not all pixels were assigned to an index ({}/{}). ".format(pixels_done, pixels)
                    +"Maybe too many segments? Extend _color_index_palette!")
    return result

def _image_color2indexed(image_xyc, expected_segments):
    # segment 0 (black) is reserved and should never be used
    # expects image in uint8 format
    if image_xyc.shape[2] != 3:
        raise ValueError("Manually edited segmentation image has invalid format. Did you add an alpha channel?")
    shape = [image_xyc.shape[0], image_xyc.shape[1]]
    result = np.zeros(shape, dtype=np.uint8)
    done = np.zeros(shape, dtype=np.bool)

    for i, color in enumerate(_color_index_palette, start=0):
        colormask = np.all(image_xyc == color, axis=-1)
        result[colormask] = i
        done[colormask] = True

    pixels = shape[0] * shape[1]
    pixels_done = np.sum(done)
    if(pixels_done != pixels):
        raise ValueError("Not all pixels could be assigned to an index ({}/{}). ".format(pixels_done, pixels)
                    +"Did you edit such that only index-colors are contained in image?")

    return result

def _calc_model_segmentation(image_file, shape, models, likelihoods, model2id, min_object_size=75):
    segmentation = np.zeros(shape, dtype=np.uint8)
    for m in models:
        modelmask = np.ones(shape, dtype=bool)
        for other in models:
            if m is not other:
                not_member = likelihoods[m] < likelihoods[other]
                modelmask[not_member] = False
        segmentation[ modelmask ] = model2id[m]

    # small object removal to hide small artifacts from each segment
    for m in models:
        mask = (segmentation == model2id[m])
        mask_fixes = np.logical_not(
                         morphology.remove_small_objects(
                             np.logical_not(mask), min_size=min_object_size ) )
        #mask_fixes[mask] = False
        segmentation[ mask_fixes ] = model2id[m]

    # save segmentation
    segment_file = filenames.segment_file(image_file)
    try:
        io.imsave(segment_file, _image_indexed2color(segmentation))
    except ValueError as e:
        raise ValueError("ERROR when trying to calculate segmentation for {}: {}".format(segment_file, e))

    return segmentation

def _load_model_segmentation(image_file, shape, models, min_object_size=75):
    segment_file = filenames.segment_file(image_file)
    try:
        segmentation = _image_color2indexed(
                            io.imread(segment_file),
                            len(models)
                        )
    except ValueError as e:
        raise ValueError("ERROR when trying to load segmentation for {}: {}".format(segment_file, e))

    return segmentation

def _count_model_segments(image_file, models, model2id, segmentation):
    # result[] will contain:
    #   filename: string, filename of image
    #   segment_[model]_pixels: number of pixels that belong most likely to this model
    #   segment_[model]_precent: area that belongs most likely to this model
    result = {}
    for m in models:
        model_mask = (segmentation == model2id[m])
        pixels = np.sum(model_mask)
        percent = (100. * pixels / (model_mask.shape[0] * model_mask.shape[1]))
        result["filename"] = image_file
        result["segment_"+m+"_area_pixel"]   = pixels
        result["segment_"+m+"_area_percent"] = percent

    return result

def _plot_model_segments(image_file, rgb_image, models, model2id, segmentation, result):
    f, axes = plt.subplots(1+len(models), 2, figsize=(20,8*(1+len(models))))
    f.suptitle(image_file, fontsize=20)
    (axis_org, axis_segments) = axes[0]
    axis_org.axis('off')
    axis_org.imshow(rgb_image)
    axis_org.set_title("original image", fontsize=16)
    axis_segments.axis('off')
    axis_segments.imshow(segmentation)
    axis_segments.set_title("all segments", fontsize=16)
    black_pixel = np.array([0,0,0], dtype=np.uint8)
    for (axis_mask, axis_maskedimage), m in zip(axes[1:], models):
        mask = (segmentation == model2id[m])
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
    figure_file = filenames.analysis_file(image_file)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figure_file, facecolor="grey", edgecolor="black")


class apply_models():
    def __init__(self, path):
        super().__init__()

        self.load_edited_segmentation = False
        self.max_threads = 3 # more than this may hog your memory. assume 2GB per thread.

        self.path = path

        self.models = collections.OrderedDict()
        for modelfile in glob.glob(filenames.model_glob(self.path)):
            model_name = filenames.modelname_from_file(modelfile)
            print("Loading model '{}'.".format(model_name))
            self.models[model_name] = pickle.load( open(modelfile, "rb") )

        self.all_images = glob.glob(filenames.preprocessed_glob(self.path))

        print("Selected {} images from '{}' to segment.".format(
                                len(self.all_images),
                                self.path
                            ))

    def apply_model_to_image(self, imagefilename):
        shape, rgb_image, points10d = _get_preprocessed_image(imagefilename)

        model2id = {m: i for i, m in enumerate(self.models, start=1)}

        segmentation_valid = False

        if self.load_edited_segmentation:
            try:
                segmentation = _load_model_segmentation(imagefilename, shape, self.models)
                segmentation_valid = True
            except FileNotFoundError as e:
                print("\n{}".format(e))
                print("RECOVERING segmentation from model. You may want to RE-EDIT this file!")

        if not segmentation_valid:
            likelihoods = _calc_model_likelihoods(shape, points10d, self.models)
            segmentation = _calc_model_segmentation(imagefilename, shape, self.models, likelihoods, model2id)

        result = _count_model_segments(imagefilename, self.models, model2id, segmentation)

        _plot_model_segments(imagefilename, rgb_image, self.models, model2id, segmentation, result)

        return result

    def run(self):
        summary_file = filenames.summary_file(self.path)
        try:
            os.remove(summary_file)
            print("Removed old summary file '{}'.".format(summary_file))
        except FileNotFoundError:
            pass

        threads = multiprocessing.cpu_count() - 1
        if threads > self.max_threads:
            threads = self.max_threads

        with multiprocessing.Pool(threads, maxtasksperchild=1) as p:
            results = p.map(self.apply_model_to_image, self.all_images, chunksize=1)

        csv = "filename;{}\n".format(";".join(("{} area [pixels];{} area [%]".format(m,m) for m in self.models)))
        for result in results:
            csv += "{}".format(result["filename"])
            for m in self.models:
                csv += ";{};{:2.4f}".format(result["segment_"+m+"_area_pixel"],
                                     result["segment_"+m+"_area_percent"])
            csv += "\n"
        with open(summary_file, "w") as csv_file:
            csv_file.write(csv)


class reapply_models(apply_models):
    def __init__(self, path):
        super().__init__(path)

        self.load_edited_segmentation = True
        self.max_threads = 6

