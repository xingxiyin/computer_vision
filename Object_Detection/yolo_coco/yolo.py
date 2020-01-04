import numpy as np
import argparse
import time
import cv2
import os


# Contruct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the input image")
ap.add_argument("-y", "--yolo", required=True,
                help="Base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="Threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


# Loading the COCO class labels our YOLO model was trained on
labelPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# Initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="unit8")

# Derive the paths to the YOLO weights and model configuration
weightPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


# Loading the YOLO object detector trained on COCO dataset(80 classes)
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)


# Loading input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(height, width) = image.shape[:2]


# Determine only the 'output' layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutputLayers()]


# Construct a blob from the input image and then perform a forward pass of the
# YOLO object deterctor, giving us our bounding boxes and associated probabilityes
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()


# Show timing information on YOLO
print("[INFO] YOLO took {:.6f} second".format(end - start))


# Initialize our lists of detected bounding boxes, confidence, and Class IDs, respectively
boxes = []
confidences = []
classIDs = []

# Loop over each of the layer outputs
for output in layerOutputs:
    # Loop over each of the detections
    for detection in output:
        # Extract the class ID and confidence(probability) of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # Filter out weak predictions by ensuring the detected probability is greater than the
        # minimum probability
        if confidence > args["confidence"]:
            # Scale the bounding box coordinates back relative to the size of image, keeping in mind that
            # YOLO actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, width, height) = box.astype("int")

            # Using the center (x, y)-coordinates to deriver the top and left corner of the bounding box
            x = int(centerX - (width/2))
            y = int(cnetrtY - (height/2))

            # Update our list of bounding box coordinates, confidence, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"]

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)