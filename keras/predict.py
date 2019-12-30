from keras.models import load_model
import argparse
import pickle
import cv2

def main(args):
    # Loading the input image and resize it to the target spatial dimensions
    image = cv2.imread(args["image"])
    output_image = image.copy()
    image = cv2.resize(image, (args["width"], args["height"]))

    # Scaling the image's raw pixel to [0, 1]
    image = image.astype("float")/255.0

    # Check to see if we should flatten the image and add a batch dimension
    if args["flatten"] != -1:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
    else:
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Loading the trainned model and label binarizer
    print("[INFO] Loading the model and label binarizer...")
    model = load_model(args["model"])
    labeler = pickle.loads(open(args["label"], "rb").read())

    # Make a prediction on the image
    preds = model.predict(image)

    # Find the class label index with the larges corresponding probability
    class_index = preds.argmax(axis=1)[0]
    label = labeler.classes_[class_index]


    # Draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][class_index] * 100)
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Image", output_image)
    cv2.waitKey(0)



if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path of the input images we are going to classify")
    ap.add_argument("-m", "--model", required=True,
                    help="Path of the trained model")
    ap.add_argument("-l", "--label", required=True,
                    help="Path of the label binarizer")
    ap.add_argument("-w", "--width", type=int, default=28,
                    help="Target spatial dimension width")
    ap.add_argument("-e", "--height", type=int, default=28,
                    help="Target spatial dimension height")
    ap.add_argument("-f", "--flatten", type=int, default=-1,
                    help="Whether or not we should flatten the image")
    args = vars(ap.parse_args())

    main(args)


    """
    python3 predict.py --image ./test/panda.jpg --model ./model/small_vgg.model --label ./label/small_vgg_lb.pickle --width 64 --height 64 --flatten -1
    
    """