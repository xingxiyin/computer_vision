import numpy as np

# non_max suppression
def non_max_suppression_slow(boxes, overlapThresh):
    """
    The algorithm assumes that the coordinates are in the following order: (x-coordinate of the top-left, y-coordinate of the top-left, x-coordinate of the bottom-right, and y-coordinate of the bottom right)

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
    # print("idxs:", idxs)

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
            xx1 = max(x1[index], x1[j])
            yy1 = max(y1[index], y1[j])
            xx2 = min(x2[index], x2[j])
            yy2 = min(y2[index], y2[j])

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



def non_max_suppression_fast(boxes, overlapThresh):
    """
    The algorithm assumes that the coordinates are in the following order: (x-coordinate of the top-left, y-coordinate of the top-left, x-coordinate of the bottom-right, and y-coordinate of the bottom right)

    :param boxes: A set of bouding boxes in the form of (startX, startY, endX, endY)
    :param overlapThresh: Overlap threshold
    :return:
    """
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []


    # If the bounding boxes integers, convert them into floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the  bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while same indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        index = idxs[last]
        pick.append(index)

        # Find the largest (x, y) coordinates for the start of the bounding box and
        # the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[index], x1[idxs[:last]])
        yy1 = np.maximum(y1[index], y1[idxs[:last]])
        xx2 = np.minimum(x2[index], x2[idxs[:last]])
        yy2 = np.minimum(y2[index], y2[idxs[:last]])

        # Compute the width and height of the bouning box
        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ration of overlap
        overlap = (width * height) / area[idxs[:last]]

        # Delete all indexes from the index list that are in the supression list
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype(int)