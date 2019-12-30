from keras.applications import ResNet50
from keras.models import load_model
from small_vggnet import SmallVGGNet
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# Initialize our Flask application and Keras model
app = flask.Flask(__name__)
model = None


def model(model_path):
    # Loading the pre-trained Keras model
    global model
    model = load_model(model_path)


def prepare_image(image, target=(64, 64)):
    # If the image is not RGB, then we convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # Return the precessed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will returned from the view
    data = {"sucess":False}

    # Ensure an image was porperly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = prepare_image(image)
            print(image.shape)

            # Classify the image and then initialize the list of predictions to return to the client
            preds = model.predict(image)
            print(preds)

            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # Loop over the results and add them to the list of returned predictions
            for (imagenetID, label, prob) in results[0]:
                result = {"label":label, probability:float(prob)}
                data["prediction"].append(result)

            # Indicate the request was a sucess
            data["success"] = True

    # Return the data dictionary as a JSON response
    return flask.jsonify(data)


def main():
    # model path
    model_path = "./model/small_vgg.model"

    # Loading the model
    model(model_path)

    # Run the application
    app.run()


if __name__ == '__main__':
    main()