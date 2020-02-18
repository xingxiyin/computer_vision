#-*-coding:utf-8-*-
import matplotlib
matplotlib.use("Agg")

from configs import data_config
from preprocessor import MeanPreprocessor
from preprocessor import PatchPreprocessor
from preprocessor import SimplePreprocessor
from preprocessor import ImageToArrayPreprocessor
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from alexnet import AlexNet
from trainingmonitor import TrainingMonitor
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import json
import os


# Construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# Loading the RGB means for the training set
means = json.loads(open(data_config.DATASET_MEAN).read())

# Initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(dbPath=data_config.TRAIN_HDF5,
                                batchSize=128,
                                preprocessor=[pp, mp, iap],
                                aug=None,
                                classes=2)
valGen = HDF5DatasetGenerator(dbPath=data_config.VAL_HDF5,
                              batchSize=128,
                              preprocessor=[sp, mp, iap],
                              classes=2)

# Initialize the optimizer
print("[INFO] compiling model..,")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct the set of callbacks
path = os.path.sep.join([data_config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages // 128,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // 128,
                    epochs=75,
                    max_queue_size=128 * 2,
                    callbacks=None, verbose=1)

# Save the model to file
print("[INFO] Serializing model...")
model.save(data_config.MODEL_PATH, overwrite=True)

# Close the HDF6 datasets
trainGen.close()
valGen.close()
