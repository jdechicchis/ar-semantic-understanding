"""
Based off of the TensorFlow image segmentation example
"""

import sys
import json
from PIL import Image
import numpy as np
from comet_ml import Experiment
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

# Comet setup
EXPERIMENT = Experiment(api_key="rHMyNdGv3wJ1sVO0v1OjvlmCo",
                        project_name="ar-semantic-understanding", workspace="jdechicchis")

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20

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
        return image, label

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

def main():
    """
    Model setup and training.
    """
    checkpoint_filepath = "./model/weights.best.hdf5"

    image_file = Image.open("./data/images/" + "00000" + ".jpg")
    image = np.asarray(image_file)
    image = image / 255.0
    annotation_file = open("./data/annotations/" + "00000" + ".json", "r")
    annotation_data = json.load(annotation_file)
    label = np.array(annotation_data["annotation"])
    image_file.close()
    annotation_file.close()
    mask = np.zeros((224, 224, 1), dtype=np.int32)
    for w in range(0, 224):
        for h in range(0, 224):
            mask[h][w] = [label[w][h]]

    model = unet_model(NUM_CLASSES)

    model.load_weights(checkpoint_filepath)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #tf.keras.utils.plot_model(model, show_shapes=True)

    pred_mask = model.predict(np.stack([image], axis=0))
    pred_mask = create_mask(pred_mask)

    pred_mask_new = np.zeros((224, 224, 1), dtype=np.int32)
    for w in range(0, 224):
        for h in range(0, 224):
            pred_mask_new[h][w] = [pred_mask[w][h]]
    #display([image, mask, pred_mask_new])

    train_test_file = open("./data/train_test_data_split.json", "r")
    train_test_split = json.load(train_test_file)

    training_generator = DataGenerator(BATCH_SIZE, train_test_split["train"])
    validation_generator = DataGenerator(BATCH_SIZE, train_test_split["test"])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks_list = [checkpoint]

    EXPERIMENT.log_parameters({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    })

    with EXPERIMENT.train():
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
