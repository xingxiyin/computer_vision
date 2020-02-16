#-*-coding:utf-8-*-
# Define the paths to the images directory
IMAGES_PATH = "/home/xingxi/Documents/learning/CV/data/cat_vs_dog/train"

# Since we don't have validation data or access to the testing labels we
# need to take a number of images from the training dataset and use them instead
NUM_CLASSES = 2
NUM_VAL_IMAGES = NUM_CLASSES * 1500
NUM_TEST_IMAGES = NUM_CLASSES * 1500

# Define te path to the ouput training, validation, and testing HDF5 files
TRIAN_HDF5 = "./home/xingxi/Documents/learning/CV/data/cat_vs_dog/hdf5/train.hdf5"
VAL_HDF5 = "./home/xingxi/Documents/learning/CV/data/cat_vs_dog/hdf5/val.hdf5"
TEST_HDF5 = "./home/xingxi/Documents/learning/CV/data/cat_vs_dog/hdf5/test.hdf5"

# Path to teh output model file
MODEL_PATH = "output/alexnet_model"

# Define the path to the dataet mean
DATASET_MEAN = "ouput/data_mean.json"

# Define the path to the output directory used for stroing plots
# classification reports, etc
OUTPUT_PATH = "../output"



