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

from models.segnet import segnet_model

from data_utils import horizontal_flip, rotate, horizontal_shear

from metrics import weighted_categorical_crossentropy
from metrics import balanced_accuracy, f1_score

# Model
MODEL = segnet_model

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

# Training parameters
BATCH_SIZE = 2
EPOCHS = 60
CLASS_WEIGHTS = [
    0.14385866807,
    0.90557223126,
    0.72059989577,
    0.66989873421,
    3.53100851804,
    2.53715235032,
    1.11641301038,
    1.18609654356
]

DATA_MEAN = [0.491024, 0.455375, 0.427466]
DATA_STD = [0.262995, 0.267877, 0.270293]

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator to provide data for training/testing.
    """
    def __init__(self, batch_size, data_ids, data_path, normalize, log=False):
        self.__batch_size = batch_size
        self.__data_ids = data_ids
        self.__current_index = 0
        self.__images_path = os.path.join(data_path, "images")
        self.__annotations_path = os.path.join(data_path, "annotations")
        self.__normalize = normalize
        self.__log = log

        self.__mean = np.array(DATA_MEAN)
        self.__std = np.array(DATA_STD)

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
        if self.__log:
            print("DataGenerator __getitem__ for index: ", index)
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

        annotation_file = open(os.path.join(self.__annotations_path, data_id + ".json"), "r")
        annotation_data = json.load(annotation_file)
        label = np.array(annotation_data["annotation"], dtype=np.uint8)

        image_file.close()
        annotation_file.close()

        image = Image.fromarray(image)
        label = Image.fromarray(label)

        # Randomly horizontal flip
        if bool(random.getrandbits(1)):
            image, label = horizontal_flip(image, label)

        if bool(random.getrandbits(1)):
            # Randomly rotate
            angle = random.uniform(-20.0, 20.0)
            image, label = rotate(image, label, angle)
        elif bool(random.getrandbits(1)):
            # Randomly horizontally shear
            amount = angle = random.uniform(-0.2, 0.2)
            image, label = horizontal_shear(image, label, amount)

        image = np.array(image)
        label = np.array(label)

        image = image / 255.0
        if self.__normalize:
            image = (image - self.__mean) / self.__std

        new_label = np.zeros((HEIGHT, WIDTH, NUM_CLASSES), dtype=np.float32)

        for h in range(0, HEIGHT):
            for w in range(0, WIDTH):
                new_label[h][w][label[h][w]] = 1

        return image, new_label

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
    parser.add_argument("--normalize", action="store_true", help="Normalize input.")
    args = parser.parse_args()

    if args.use_comet:
        experiment = Experiment(api_key="rHMyNdGv3wJ1sVO0v1OjvlmCo",
                                project_name="ar-semantic-understanding", workspace="jdechicchis")

    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("GPUs: ", gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        with tf.device("/gpu:0"):
            model = MODEL(WIDTH, HEIGHT, NUM_CLASSES)
    else:
        with tf.device("/cpu:0"):
            model = MODEL(WIDTH, HEIGHT, NUM_CLASSES)

    if args.load_weights:
        model.load_weights(args.checkpoint_file)

    # "accuracy" is CategoricalAccuracy here
    model.compile(optimizer="adam",
                  loss=weighted_categorical_crossentropy(CLASS_WEIGHTS),
                  metrics=["accuracy",
                           balanced_accuracy(),
                           tf.keras.metrics.AUC(),
                           tf.keras.metrics.MeanIoU(num_classes=8),
                           f1_score()])

    if args.plot_model:
        tf.keras.utils.plot_model(model, show_shapes=True)

    train_test_file = open(os.path.join(args.data_path, "train_test_data_split.json"), "r")
    train_test_split = json.load(train_test_file)

    training_generator = DataGenerator(BATCH_SIZE, train_test_split["train"], args.data_path, args.normalize)
    validation_generator = DataGenerator(BATCH_SIZE, train_test_split["test"], args.data_path, args.normalize)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.checkpoint_file,
                                                    monitor="val_accuracy",
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode="max")
    callbacks_list = [checkpoint]

    def model_fit():
        model.fit(x=training_generator,
                  epochs=EPOCHS,
                  steps_per_epoch=len(train_test_split["train"]) // BATCH_SIZE,
                  validation_steps=len(train_test_split["test"]) // BATCH_SIZE,
                  validation_data=validation_generator,
                  callbacks=callbacks_list)

    if args.use_comet:
        experiment.log_parameters({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "class_weights": CLASS_WEIGHTS,
            "normalize": args.normalize
        })

        with experiment.train():
            model_fit()
    else:
        model_fit()

if __name__ == "__main__":
    sys.exit(main())
