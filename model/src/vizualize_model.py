"""
Visualize model outputs.
"""

import sys
import os
import argparse
import json
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from models.unet import unet_model

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

def display(display_list):
    """
    Display image, true mask, and predicted mask.
    """
    plt.figure(figsize=(15, 5))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i, display_item in enumerate(display_list):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_item))
        plt.axis("off")

    close_button_axis = plt.axes([0.7, 0.05, 0.1, 0.075])
    close_button = Button(close_button_axis, "Close")

    def close_plot(_):
        """
        Close the plot.
        """
        plt.close()

    close_button.on_clicked(close_plot)

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
    print(
        "#####################################\n"
        "# WARNING: May not work with TF GPU #\n"
        "#####################################\n\n"
    )

    parser = argparse.ArgumentParser(description="Train model on SUN RGB-D data.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file.")
    args = parser.parse_args()

    with tf.device("/cpu:0"):
        model = unet_model(WIDTH, HEIGHT, NUM_CLASSES)

    if args.checkpoint_file:
        model.load_weights(args.checkpoint_file)

    images_path = os.path.join(args.data_path, "images")
    annotations_path = os.path.join(args.data_path, "annotations")

    data_ids_array = [f"{i:05}" for i in range(0, 10335)]
    random.shuffle(data_ids_array)
    for data_id in data_ids_array:
        image_file = Image.open(os.path.join(images_path, data_id + ".jpg"))
        image = np.asarray(image_file)
        image = image / 255.0
        annotation_file = open(os.path.join(annotations_path, data_id + ".json"), "r")
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

if __name__ == "__main__":
    sys.exit(main())
