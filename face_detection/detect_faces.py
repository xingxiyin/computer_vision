import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="Path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")


# Load our serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Loading the input image and construct an input blob for the image
# by resizing to a fixed 300*300 pixels and then normalizing it
image = cv2.imread(args["image"])
(height, weight) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections =net.fotward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence(i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
    if confidence > args["confidence"]:
        # Compute the (x, y)-coordinates of bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([weight, height, weight, height])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bouding box of the face along with the associated probability
        text = "{:.2}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startY, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255, 2))

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)