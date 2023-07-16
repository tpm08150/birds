import anvil.server
import anvil.media
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import json

anvil.server.connect("server_3PL4U47LLHCMGS6SPHWDOMSJ-UO6EW4OV2QOWW3JA")

# Load your trained model
model = tf.keras.models.load_model('/Users/tylermorton/PycharmProjects/birds_dataset/my_model/')


def preprocess_image(img, target_size):
    """Preprocess the image for model prediction.

    This function should implement the same preprocessing steps
    as you used during model training.
    """
    # Resize the image to the target size
    img = img.resize(target_size)

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Expand dimensions to be (1, height, width, channels)
    # This is because the model expects batches of images
    img_array = np.expand_dims(img_array, axis=0)

    # Scale the image pixels
    img_array /= 255.

    return img_array


@anvil.server.callable
def predict_bird_species(image):
    # Convert the Anvil Media object to a PIL Image
    with anvil.media.TempFile(image) as filename:
        img = Image.open(filename)

    # Preprocess your image and make prediction
    img_size = 224  # Make sure to set this to the same size as used in training
    preprocessed_img = preprocess_image(img, (img_size, img_size))
    prediction = model.predict(preprocessed_img)

    # Get the index of the maximum value
    index = np.argmax(prediction)

    # Get the indices of the top 3 maximum values
    top3_indices = prediction[0].argsort()[-3:][::-1]

    # Load class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Invert the dictionary
    index_class = {v: k for k, v in class_indices.items()}

    # Find the classes that correspond to the top 3 highest probabilities
    top3_species = [(index_class[i], prediction[0][i]) for i in top3_indices]

    # Format output
    result = ', '.join(f"{species} - {confidence * 100:.2f}% confidence, "
                       f"Google it: https://www.google.com/search?q={species.replace(' ', '+')}"
                       for species, confidence in top3_species)

    return result


anvil.server.wait_forever()

anvil.server.wait_forever()
