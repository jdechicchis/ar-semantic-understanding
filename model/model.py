"""
Based off of the TensorFlow image segmentation example
"""

import sys
import json
import random
from PIL import Image
import numpy as np
import scipy
from comet_ml import Experiment
import tensorflow as tf
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

# Comet setup
EXPERIMENT = None
#EXPERIMENT = Experiment(api_key="rHMyNdGv3wJ1sVO0v1OjvlmCo",
#                        project_name="ar-semantic-understanding", workspace="jdechicchis")

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
CLASS_WEIGHTS_ARRAY = [1, 175, 8, 9, 137, 140, 34, 44]
CLASS_WEIGHTS = K.variable(np.array(CLASS_WEIGHTS_ARRAY))

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator to provide data for training/testing.
    """
    def __init__(self, batch_size, data_ids):
        self.__batch_size = batch_size
        self.__data_ids = data_ids
        self.__current_index = 0

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
        image_file = Image.open("./data/images/" + data_id + ".jpg")
        image = np.asarray(image_file)
        image = image / 255.0
        annotation_file = open("./data/annotations/" + data_id + ".json", "r")
        annotation_data = json.load(annotation_file)
        label = np.array(annotation_data["annotation"])
        image_file.close()
        annotation_file.close()

        # Randomly horizontal flip
        if bool(random.getrandbits(1)):
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        new_label = np.zeros((HEIGHT, WIDTH, NUM_CLASSES), dtype=np.float32)

        for h in range(0, HEIGHT):
            for w in range(0, WIDTH):
                new_label[h][w][label[h][w]] = 1

        return image, new_label

def unet_model(output_channels):
    """
    UNet-based segmentation model.
    """
    # Use MobileNetV2 as the encoder
    base_model = tf.keras.applications.MobileNetV2(input_shape=[WIDTH, HEIGHT, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[WIDTH, HEIGHT, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up_layer, skip in zip(up_stack, skips):
        x = up_layer(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

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

def custom_f1_score(y_true, y_pred):
    """
    Calculate f1-score.
    """
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def weighted_categorical_crossentropy(y_true, y_pred):
    """
    Calculated weighted categorical crossentropy.
    """
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * CLASS_WEIGHTS
    loss = -K.sum(loss, -1)
    return loss

def calculate_recall(y_true, y_pred):
    """
    Calculate the recall (i.e. true positive rate).
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def calculate_true_negative_rate(y_true, y_pred):
    """
    Calculate the true negative rate.
    """
    #TODO: implement
    #print(y_true.shape)
    true_negatives = K.sum(K.round(-1 * K.clip(y_true * y_pred, 0, 1) + 1))
    possible_negatives = (BATCH_SIZE * HEIGHT * WIDTH * NUM_CLASSES) -  K.sum(K.round(K.clip(y_true, 0, 1)))
    true_negative_rate = true_negatives / (possible_negatives + K.epsilon())
    #print("\n\ntrue_negatives: {}\npossible_negatives: {}\ntrue_negative_rate: {}\n\n".format(true_negatives, possible_negatives, true_negative_rate))
    return true_negative_rate

def calculate_precision(y_true, y_pred):
    """
    Calculate precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    """
    Calculate f1-score.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced_accuracy.
    """
    #TODO: implement
    #y_true_new = K.argmax(y_true)
    #y_pred_new = K.argmax(y_pred)
    #true_positives = K.sum(y_true_new == y_pred_new)
    #true_negatives = K.sum(y_true_new != y_pred_new)
    #print(y_true)
    #print(y_true.numpy())
    #true_positive_rate = calculate_recall(y_true, y_pred)
    #true_negative_rate = calculate_true_negative_rate(y_true, y_pred)
    #print("true_positive_rate: {}".format(true_positive_rate))
    #print("true_negative_rate: {}".format(true_negative_rate))
    #t = K.eval(y_true)
    #print(np.shape(t))
    return 0#(true_positive_rate + true_negative_rate) / 2

def iou(y_true, y_pred):
    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    """
    intersection = K.round(K.clip(y_true * y_pred, 0, 1))
    union = K.round(K.clip(y_true + y_pred, 0, 1))
    return K.sum(intersection) / K.sum(union)

def main():
    """
    Model setup and training.
    """
    checkpoint_filepath = "./checkpoints/weights.best.hdf5"

    """
    print(np.shape(image))
    image = scipy.ndimage.rotate(image, angle=10, mode="reflect", reshape=False)
    print(np.shape(image))
    print(np.shape(label))
    label = scipy.ndimage.rotate(label, angle=-10, mode="reflect", reshape=False)
    print(np.shape(label))
    """

    model = unet_model(NUM_CLASSES)

    model.load_weights("./checkpoints/experiment_5de153015/weights.best.hdf5")

    model.compile(optimizer="adam",
                  loss=weighted_categorical_crossentropy,#"categorical_crossentropy", #sparse_categorical_crossentropy
                  metrics=["accuracy", "CategoricalAccuracy", custom_f1_score, f1_score, balanced_accuracy, iou])
                  #weighted_metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)

    #00004
    #07400
    data_ids_array = [f"{i:05}" for i in range(0, 10335)]
    random.shuffle(data_ids_array)
    for data_id in data_ids_array:
        image_file = Image.open("./data/images/" + data_id + ".jpg")
        image = np.asarray(image_file)
        image = image / 255.0
        annotation_file = open("./data/annotations/" + data_id + ".json", "r")
        annotation_data = json.load(annotation_file)
        label = np.array(annotation_data["annotation"])
        image_file.close()
        annotation_file.close()
        mask = np.zeros((224, 224, 1), dtype=np.int32)

        for w in range(0, 224):
            for h in range(0, 224):
                mask[h][w] = [label[h][w]]

        pred_mask = model.predict(np.stack([image], axis=0))
        pred_mask = create_mask(pred_mask)

        pred_mask_new = np.zeros((224, 224, 1), dtype=np.int32)
        for h in range(0, 224):
            for w in range(0, 224):
                pred_mask_new[h][w] = [pred_mask[h][w]]
        display([image, mask, pred_mask_new])

    train_test_file = open("./data/train_test_data_split.json", "r")
    train_test_split = json.load(train_test_file)

    training_generator = DataGenerator(BATCH_SIZE, train_test_split["train"])
    validation_generator = DataGenerator(BATCH_SIZE, train_test_split["test"])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                    monitor="val_f1_score",
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode="max")
    callbacks_list = [checkpoint]

    """
    if EXPERIMENT:
        EXPERIMENT.log_parameters({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "class_weights": CLASS_WEIGHTS_ARRAY
        })

        with EXPERIMENT.train():
            """
    model.fit_generator(generator=training_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=len(train_test_split["train"]) // BATCH_SIZE,
                        validation_steps=len(train_test_split["test"]) // BATCH_SIZE,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=callbacks_list)

if __name__ == "__main__":
    sys.exit(main())
