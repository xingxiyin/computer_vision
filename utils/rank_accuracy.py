#-*-coding:utf-8-*-
import argparse
import pickle
import h5py
import numpy as np

def rank5_accuracy(preds, labels):
    """

    :param preds: An N*T matrix, N is the number of rows, contains the probabilies associated with
                    with each class label T
    :param lables: The ground-truth labels for the images in the dataset
    :return:
    """
    # Initialize the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0

    # Loop over the predictions and ground-truth labels
    for (pred, label) in zip(preds, labels):
        # Sort the probabilites in the descending order
        pred_index = np.argsort(pred)[::-1]

        # Check if the ground-truth label is in the top-5 predictions
        if label in pred[:5]:
            rank5 += 1

        # Ckeck to see if the groud-truth is the top-1 prediction
        if label == pred[0]:
            rank1 += 1

    # Compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    # return the tuple of the rank-1 and rank5 accuracies
    return (rank1, rank5)

def main(args):
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

if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--db", required=True, help="Path HDFS database")
    ap.add_argument("-m", "--model", required=True, help="Path to the pre-trained model")
    args = vars(ap.parse_args())

    # Runing the main function
    main(args)