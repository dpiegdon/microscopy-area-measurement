# fdm=indent foldlevel=1 foldnestmax=2

import os

def raw_image_glob(basedir):
    return "{}/*.tif".format(basedir)

def modeldir_glob(basedir):
    return "{}/model_*".format(basedir)

def model_file(basedir, modelname):
    return "{}/model-{}.gmmpickle".format(basedir, modelname)

def modelname_from_file(modelfile):
    return modelfile.split("model-")[1].replace(".gmmpickle", "")

def model_glob(basedir):
    return model_file(basedir, "*")

def preprocessed_file(base_image_filename):
    return str(base_image_filename).replace(
                                    ".tif", "_preprocessed.png")

def preprocessed_glob(basedir):
    return "{}/*_preprocessed.png".format(basedir)

def analysis_glob(basedir):
    return "{}/*_analysis.png".format(basedir)

def segment_glob(basedir):
    return "{}/*_segmentation.png".format(basedir)

def segment_file(preprocessed_image_file):
    return str(preprocessed_image_file).replace(
                                    "_preprocessed.png", "_segmentation.png")

def summary_file(path):
    return "{}/analysis-summary.csv".format(path)

def analysis_file(preprocessed_image_file):
    return str(preprocessed_image_file).replace(
                                    "_preprocessed.png", "_analysis.png")

