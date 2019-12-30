import config
from imutils import paths
import shutil
import os

# Loop over the dataset
for split in (config.TRAIN, config.TEST, config.VAL):
    # Grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    path = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
    imagePaths = list(paths.list_images(path))

    # Loop over the image path
    for imagePath in imagePaths:
        # Extract class label from the filename
        filename = imagePaths.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])]

        # Construct the path to the output directory
        dirPath = os.path.sep.join([config.BASE_PATH, split, label])

        # If the output directory does not exit, create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Construct the path to the output image file and copy it
        path = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, path)











































