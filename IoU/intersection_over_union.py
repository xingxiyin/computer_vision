
def IoU(gtBox, predBox):
    """

    :param gtBox: the ground-truth bounding box
    :param predBox: the predicted bounding box
    :return: intersection over union
    """
    # Determine the (x, y)-coordinate of the intersection rectangle
    x_gtBox = max(gtBox[0], predBox[0])
    y_gtBox = max(gtBox[1], predBox[1])
    x_predBox = min(gtBox[2], predBox[2])
    y_predBox = min(gtBox[3], predBox[3])

    # Compute the area of intersection rectangle
    interArea = max(0, x_predBox - x_gtBox + 1) * max(0, y_predBox - y_gtBox + 1)

    # Compute the area of both the prediction and ground-truth rectangle
    boxGtArea = (gtBox[2] - gtBox[0] + 1) * (gtBox[3] - gtBox[0] + 1)
    boxPredArea = (predBox[2] - predBox[0] + 1) * (predBox[3] - predBox[0] + 1)

    # Compute the intersection over union
    iou = interArea /float(boxGtArea + boxPredArea - interArea)

    # Return the IOU value
    return iou
