"""
Train a model on the SUN RGB-D dataset.
"""

import sys
import os
import argparse
import json
import random
from PIL import Image
import numpy as np
from comet_ml import Experiment
import tensorflow as tf
import matplotlib.pyplot as plt

from models.unet import unet_model

from metrics import weighted_categorical_crossentropy
from metrics import custom_f1_score

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
CLASS_WEIGHTS = [1, 175, 8, 9, 137, 140, 34, 44]

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator to provide data for training/testing.
    """
    def __init__(self, batch_size, data_ids, data_path):
        self.__batch_size = batch_size
        self.__data_ids = data_ids
        self.__current_index = 0
        self.__images_path = os.path.join(data_path, "images")
        self.__annotations_path = os.path.join(data_path, "annotations")

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return len(self.__data_ids) // self.__batch_size

    def __getitem__(self, index):
        """
        Return one batch of data
        """
        data = []
        labels = []
        for _ in range(0, self.__batch_size):
            sample, label = self.__load_data(self.__data_ids[self.__current_index])
            data.append(sample)
            labels.append(label)
            self.__current_index += 1
            if self.__current_index > len(self.__data_ids) - 1:
                self.__current_index = 0
        return np.stack(data, axis=0), np.stack(labels, axis=0)

    def __load_data(self, data_id):
        """
        Return data sample and label.
        """
        image_file = Image.open(os.path.join(self.__images_path, data_id + ".jpg"))
        image = np.asarray(image_file)
        image = image / 255.0
        annotation_file = open(os.path.join(self.__annotations_path, data_id + ".json"), "r")
        annotation_data = json.load(annotation_file)
        label = np.array(annotation_data["annotation"])
        image_file.close()
        annotation_file.close()

        # Randomly horizontal flip
        if bool(random.getrandbits(1)):
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        # TODO: Add crop
        # TODO: Add shear
        # TODO: Add rotate

        new_label = np.zeros((HEIGHT, WIDTH, NUM_CLASSES), dtype=np.float32)

        for h in range(0, HEIGHT):
            for w in range(0, WIDTH):
                new_label[h][w][label[h][w]] = 1

        return image, new_label

def display(display_list):
    """
    Display image, true mask, and predicted mask.
    """
    plt.figure(figsize=(15, 5))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i, display_item in enumerate(display_list):
    #for i in range(0, len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_item))
        plt.axis('off')
    plt.show()

    should_continue = input("Do you want to continue? [y/n] ")
    if should_continue is "n":
        sys.exit(0)

def create_mask(pred_mask):
    """
    Get the mask from a prediction.
    """
    pred_mask = np.argmax(pred_mask, axis=3)
    return pred_mask[0]

def main():
    """
    Model setup and training.
    """
    parser = argparse.ArgumentParser(description="Train model on SUN RGB-D data.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("checkpoint_file", type=str, help="Checkpoint file.")
    parser.add_argument("--load_weights", action="store_true", help="Load weights from checkpoint.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--plot_model", action="store_true", help="Plot the model architecture.")
    parser.add_argument("--use_comet", action="store_true", help="Log experiment to Comet ML.")
    args = parser.parse_args()

    if args.use_comet:
        experiment = Experiment(api_key="rHMyNdGv3wJ1sVO0v1OjvlmCo",
                                project_name="ar-semantic-understanding", workspace="jdechicchis")

    if args.gpu:
        with tf.device("/gpu:0"):
            model = unet_model(WIDTH, HEIGHT, NUM_CLASSES)
    else:
        with tf.device("/cpu:0"):
            model = unet_model(WIDTH, HEIGHT, NUM_CLASSES)

    if args.load_weights:
        model.load_weights(args.checkpoint_file)

    model.compile(optimizer="adam",
                  loss=weighted_categorical_crossentropy(CLASS_WEIGHTS),
                  metrics=["accuracy", "CategoricalAccuracy", custom_f1_score])

    if args.plot_model:
        tf.keras.utils.plot_model(model, show_shapes=True)

    train_test_file = open(os.path.join(args.data_path, "train_test_data_split.json"), "r")
    train_test_split = json.load(train_test_file)

    training_generator = DataGenerator(BATCH_SIZE, train_test_split["train"], args.data_path)
    validation_generator = DataGenerator(BATCH_SIZE, train_test_split["test"], args.data_path)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.checkpoint_file,
                                                    monitor="val_f1_score",
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode="max")
    callbacks_list = [checkpoint]

    def model_fit():
        model.fit_generator(generator=training_generator,
                            epochs=EPOCHS,
                            steps_per_epoch=len(train_test_split["train"]) // BATCH_SIZE,
                            validation_steps=len(train_test_split["test"]) // BATCH_SIZE,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            callbacks=callbacks_list)


    if args.use_comet:
        experiment.log_parameters({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "class_weights": CLASS_WEIGHTS
        })

        with experiment.train():
            model_fit()
    else:
        model_fit()

if __name__ == "__main__":
    sys.exit(main())
