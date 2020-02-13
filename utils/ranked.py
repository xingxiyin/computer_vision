#-*-coding:utf-8-*-
import numpy as np

def rank5_accuracy(preds, lables):
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