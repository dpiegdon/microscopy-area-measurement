# fdm=indent foldlevel=1 foldnestmax=2

import shutil
import os
import glob

from . import filenames

def clear_inconsistent_files(removelist):
    print("Removing obsolete files to avoid inconsistent data:")
    for f in removelist:
        try:
            print("  remove  "+f)
            os.remove(f)
        except FileNotFoundError:
            pass


def clear_preprocessed_images(path):
    removelist = glob.glob(filenames.preprocessed_glob(path))
    clear_inconsistent_files(removelist)

def clear_trained_models(path):
    removelist = glob.glob(filenames.model_glob(path))
    clear_inconsistent_files(removelist)

def clear_applied_results(path, keep_edited_segmentation):
    removelist = [ filenames.summary_file(path) ]
    if not keep_edited_segmentation:
        removelist += glob.glob(filenames.segment_glob(path))
    removelist += glob.glob(filenames.analysis_glob(path))
    clear_inconsistent_files(removelist)

def clear_model_definitions(path):
    removedirs = glob.glob(filenames.modeldir_glob(path))
    for r in removedirs:
        shutil.rmtree(r)

class clear_all():
    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        clear_preprocessed_images(self.path)
        clear_trained_models(self.path)
        clear_applied_results(self.path, False)

class clear_really_all(clear_all):
    def run(self):
        super().run()
        clear_model_definitions(self.path)



