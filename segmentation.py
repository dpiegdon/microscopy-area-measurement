#!/usr/bin/env python3
# fdm=indent foldlevel=1 foldnestmax=2

from assets import preprocess
from assets import train_models
from assets import apply_models

import sys
import collections

commands = collections.OrderedDict([
    ('preprocess',     (preprocess.preprocess,               "preprocess images and add alpha channels")),
    ('train',          (train_models.train_models,           "train segmentation models")),
    ('apply',          (apply_models.apply_models,           "apply segmentation models to all images")),
    ('reapply',        (apply_models.reapply_models,         "reapply segmentation according to images that have been edited by user")),
])

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in commands:
        print("required arguments: <command> <path> [path [...] ]")
        print("where command is one of:")
        for c in commands:
            print("  {} -- {}".format(c, commands[c][1]))
        sys.exit(1)

    taskname = sys.argv[1]
    paths = sys.argv[2:]

    print("processing files in these paths:")
    for path in paths:
        print("  " + path)

    rets = []
    for path in paths:
        task = commands[taskname][0](path)
        ret = task.run()
        print("task '{}' for path '{}' finished with result '{}'".format(taskname, path, ret))
        rets.append(ret)

    print("overall result for task '{}':".format(taskname))
    for path, ret in zip(paths, rets):
        print("  path {}: {}".format(path, "ok" if ret == 0 else "error"))
    ret = sum(rets)
    sys.exit(ret)

