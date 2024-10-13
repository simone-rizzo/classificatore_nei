import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1,NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

batch_size = 64
img_height = 224
img_width = 224

data_dir = "."
benign_dir = os.path.join(data_dir, "benign")
malignant_dir = os.path.join(data_dir, "malignant")

# Data augmentation e preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Divide i dati in train/validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Cambiato a 'categorical'
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Cambiato a 'categorical'
    subset='validation'
)

# Ho optato per il transfer learning
# con il modello NASNetMobile pre-addestrato
base_model = NASNetMobile(
        weights='imagenet',
        include_top=False,
        input_shape=(img_width, img_height, 3))

# Congela i pesi del modello preaddestrato
for layer in base_model.layers:
    layer.trainable = False

# Aggiungi strati personalizzati sopra il modello preaddestrato
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)  # Cambia l'output a 2 classi con softmax

model = Model(inputs=base_model.input, outputs=output)

# Compila il modello con categorical_crossentropy per multi-class classification
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Aggiungi EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Addestramento con EarlyStopping
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[early_stopping],
    shuffle=False
)

# Salva il modello
model.save('neo_classifier.h5')

# Visualizza le performance
import matplotlib.pyplot as plt

# Estrai le perdite di training e validazione
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Estrai l'accuratezza di training e validazione
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Grafico delle perdite
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Grafico dell'accuratezza
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()