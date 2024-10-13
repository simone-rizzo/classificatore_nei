import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Carica il modello salvato
model_path = 'neo_classifier.h5'
model = tf.keras.models.load_model(model_path)

# Definisci le classi
class_names = ['benign', 'malignant']

def preprocess_image(img_path, img_height=224, img_width=224):
    """Preprocessa l'immagine per il modello."""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi dimensione batch
    img_array /= 255.0  # Normalizza i valori dell'immagine
    return img_array

def make_prediction(img_path):
    """Effettua la predizione sull'immagine."""
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Estrai la classe e la confidenza
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence


img_path = "mio_neo.jpg"
predicted_class, confidence = make_prediction(img_path)
print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")
