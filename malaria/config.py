import os

# Initialize the path to the 'original' input directory of images
ORIG_INPUT_DATASET = "malaria/cell_images"

# Initialize the base path to the "new" directory that will contain our images
# after computing the trianing and testing split dataset
BASE_PATH = "malaria"

# Derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "trianing"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# Define the amount of data that will be useed for training
TRAIN_SPLIT = 0.8

# The amount of validation data will be a percentage of the training data
VAL_SPlit = 0.1