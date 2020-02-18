#-*-coding:utf-8-*-
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # Store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def prprocess(self, image):
        # Resize the image to a fixed size, ignoring the aspect ration
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # Store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # Apply the Keras utility function that correctly rearrange
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)

class MeanPreprocessor:
    """
    Mean subtraction is used to reduce the affects of lighting variations during classification
    """
    def __init__(self, rMean, gMean, bMean):
        # Store the Red, Green, and Blue channel averages across
        # a training dataset
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # Split the image into its respective Red, Green and Blue channels
        # The OpenCV represents images in BGR order rather RGB
        (B, G, R) = cv2.split(image.astype("float32"))

        # Sbustract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # Merge the channels back together and return the image
        return cv2.merge([B, G, R])


class PatchPreprocessor:
    """
    1) The PatchPreprocessor is responsible for randomly sampling M*N region of an inage
    during the training process.
    2) Apply the patch preprocessing when the spatial dimensions of the images
    are larger than what the CNN expects.
    3) patch preprocessing is similar to dataa augmentation which can reduce overfitting
    """
    def __init__(self, width, height):
        # Store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # Extract a random crop from the image with the target width and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]


class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # Store the target image width, height, whether or not horizontal
        # flips should be included, along with the interpolation method used
        # when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # Initialize the list crops
        crops = []

        # Grab the width and height of the image then use these dimenstions
        # to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],    # Top-left
            [w-self.width, 0, w, self.height],  # Top-right
            [w-self.width, h-self.height, w, h],# bottom-right
            [0, h-self.height, self.width, h]]  # bottom-left

        # Compute the center crop of the image
        delta_width = int(0.5 * (w - self.width))
        delta_height = int(0.5 * (h - self.height))
        coords.append([delta_width, delta_height, w - delta_width, h - delta_height]) # Center crop

        # Loop over the coordinates, extract each of the crops, and resize each of them to a fixe size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # Check to see if the horizontal flips should be taken
        if self.horiz:
            # Compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(crop, 1) for crop in crops]
            crops.extend(mirrors)

        # return the set of crops
        return np.array(crops)


