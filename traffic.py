import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
IMG_CHANNEL = 3
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    print("Get image arrays and labels for all image files")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    print("Split data into training and testing sets")
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print("Get a compiled neural network")
    # Get a compiled neural network
    model = get_model()

    print("Fit model on training data")
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    print("Evaluate neural network performance")
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images_list, file_labels = list(), list()
    for file_path in glob(os.path.realpath(data_dir) + "/*/*", recursive=True):
        raw_image = cv2.imread(file_path)
        images_list.append(cv2.resize(raw_image, (IMG_WIDTH, IMG_HEIGHT)))
        file_labels.append(int(file_path.split(os.sep)[-2]))
    return images_list, file_labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(units=NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    main()
