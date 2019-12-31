import numpy as np

# non_max suppression
def non_max_suppression_slow(boxes, overlapThresh):
    """

    :param boxes: A set of bouding boxes in the form of (startX, startY, endX, endY)
    :param overlapThresh: Overlap threshold
    :return:
    """
    # If box number is 0, return an empty list
    if len(boxes) == 0:
        return []

    # Initilize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the  bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # it is critical that we sort according to the bottom-right corner as we will need to compute
                           # the overlap ratio of other bounding boxes later


    # Keep looping while same indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list, add the index value to the list of picked
        # indexes, then initialize the suppression list using the last index
        last = len(idxs) - 1
        index = idxs[last]
        pick.append(index)
        suppress = [last]

        # Loop over all indexes in the indxes list
        for pos in range(0, last):
            # Grab the current index
            j = idxs[pos]

            # Find the largest (x, y) coordinates for the start of the bounding box and
            # the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the current bouning box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete all indexes from the index list that are in the supression list
        idxs = np.delete(idxs, suppress)


    # Return only the bounding boxes that were picked
    return boxes[pick]