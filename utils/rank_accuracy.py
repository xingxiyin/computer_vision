#-*-coding:utf-8-*-
from ranked import rank5_accuracy
import argparse
import pickle
import h5py

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="Path HDFS database")
ap.add_argument("-m", "--model", required=True, help="Path to the pre-trained model")
args = vars(ap.parse_args())

# Loading the pre-trained model
print("[INFO] Loading the pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())

# Open the HDF5 database for reading. Then determine the index of the
# trainning and testing split, provied that this data was  already shuffled
# *prior* to writing it to disk
db = h5py.File(args["db"], "r")
train_num = int(db["label"].shape[0] * 0.75)

# Making predictions on the testing set then compute the rank-1
# and rank-5 accuracies
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank_1, rank_5) = rank5_accuracy(preds, db["labels"][i:])
print("[INFO] Rank-1: {0.2f}%".format(rank_1*100))
print("[INFO] Rank-5: {0.2f}%".format(rank_5*100))

# Close the database
db.close()