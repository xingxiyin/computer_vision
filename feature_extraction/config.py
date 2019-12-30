import os

# Initialize the path to the 'original' input directory of images
ORIG_INPUT_DATASET = "Food-5k"

# Initialize the base path to the 'new' directory that contain
# image after computing the training and testing dataset
BASE_PATH = "dataset"

# Define the names of the training, testing and validation directories
TRAIN = "training"
VAL = "validation"
TEST = "evaluation"

# Initialize the list of class label names
CALSSES = ["non-food", 'food']

# Set the batch size
BATCH_SIZE = 32

# Initialize the label encoder file path and the output directory to
# where the extracted feature (in the CSV format) to be stroed
LABEL_PATH = os.path.sep.join(['output', 'lable.cpickle'])
BASE_CSV_PATH = "output"




