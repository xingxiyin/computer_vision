import imutils

def pyramid(image, scale=1.05, minSize=(30, 30)):
    # yield the orignal image
    yield image

    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=width)

        # If the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # Yield the next image in the pyramid
        yield image



def sliding_window(image, stepSize, windowSize):
    """

    :param image: image that we are going to loop over
    :param stepSize: indicate how many pixels we are going to slip in both (x, y) direction.
                    it's common to use a stepSize of 4 or 8 pixels.
                    The smaller of the stepsize is, the more window will need to be examine
    :param windowSize: Define the width and height of the window we are going to extract from the image
    :return:
    """
    # Slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # Yield the current window
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

