#!/usr/bin/env python3
# fdm=indent foldlevel=1 foldnestmax=2

from assets import preprocess
from assets import train_models
from assets import apply_models
from assets import clear

import sys
import functools
import traceback
import collections

import warnings

commands = collections.OrderedDict([
    ('preprocess',     (preprocess.preprocess,              "preprocess images and add alpha channels")),
    ('train',          (train_models.train_models,          "train segmentation models")),
    ('apply',          (apply_models.apply_models,          "apply segmentation models to all images")),
    ('reapply',        (apply_models.reapply_models,        "reapply segmentation according to images that have been edited by user")),
    ('clearall',       (clear.clear_all,                    "clear all derived data")),
    ('clearreallyall', (clear.clear_really_all,             "clear all derived data, including the manually prepared model definitions")),
])

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

    if len(sys.argv) < 3 or sys.argv[1] not in commands:
        print("Required arguments: <command> <path> [path [...] ]")
        print("Where command is one of:")
        for c in commands:
            print("  {} -- {}".format(c, commands[c][1]))
        sys.exit(1)

    taskname = sys.argv[1]
    paths = sys.argv[2:]

    print("Processing files in these paths:")
    for path in paths:
        print("  " + path)

    rets = collections.OrderedDict()

    for path in paths:
        print('>'*80)
        print("Task '{}' for path '{}' starting...".format(taskname, path))
        task = commands[taskname][0](path)
        try:
            task.run()
            rets[path] = True
            print("Task '{}' for path '{}' succeeded.".format(taskname, path))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exc_text = functools.reduce(lambda x,y: x+y, traceback.format_exception(exc_type, exc_value, exc_traceback))
            print("RAISED EXCEPTION: \n{}".format(exc_text))
            rets[path] = False
        print('<'*80)

    print("\n\nResult summary for task '{}':".format(taskname))
    overall_ok = True
    for path in rets:
        print("  For path '{}': {}".format(path, "ok" if rets[path] else "ERROR"))
        overall_ok &= rets[path]
    sys.exit(0 if overall_ok else -1)

