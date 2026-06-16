import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(
    "model/fruit_classifier.keras"
)

classes = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Strawberry"
]

def predict_fruit(img_path):

    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    return predicted_class, confidence