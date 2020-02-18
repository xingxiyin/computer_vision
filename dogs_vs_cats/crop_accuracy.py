#-*-coding:utf-8-*-
from configs import data_config
from preprocessor import MeanPreprocessor
from preprocessor import PatchPreprocessor
from preprocessor import CropPreprocessor
from preprocessor import SimplePreprocessor
from preprocessor import ImageToArrayPreprocessor
from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.models import load_model
import json

# Loading the RGB means for the training set
means = json.loads(open(data_config.DATASET_MEAN).read())

# Initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# Load the pretrained network
print("[INFO] Loading model")
model = load_model(data_config.MODEL_PATH)

# Initialize the testing dataset generator, then make predictions on the testing dataset
print("[INFO] Predicting on test dataset...")
testGen = HDF5DatasetGenerator(data_config.TEST_HDF5, 64,
                               preprocessor=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(),
                                      steps=testGen.numImage//64,
                                      max_queue_size=64*2)

# compute the rank-1 and rank-5 accuracies
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()