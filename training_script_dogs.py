import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import ssl
import certifi
import urlopen
import requests
import anvil.server
import json


# Define parameters
img_size = 224  # Depending on the model you choose
batch_size = 32

def training_model_birds():
    # Define your data generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load images from the disk, applies rescaling, and resizes the images
    train_generator = datagen.flow_from_directory(
        'birds_dataset',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    val_generator = datagen.flow_from_directory(
        'birds_dataset',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    # Define your model
    model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                              include_top=False,
                                              weights='imagenet')
    model.trainable = False  # Freeze the convolutional base

    # Add a new classification layer
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=10, validation_data=val_generator)

    model.save('/Users/tylermorton/PycharmProjects/birds_dataset/my_model')

    # After you've defined your generators...
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

training_model_birds()
