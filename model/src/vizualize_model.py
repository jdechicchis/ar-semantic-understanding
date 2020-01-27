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
import matplotlib.patches as patches
from matplotlib.widgets import Button

from models.unet import unet_model

# Number of classes per pixel
NUM_CLASSES = 8
CLASS_LABELS_AND_COLORS = [
    {"color": [255, 255, 255], "label": "Unknown"},
    {"color": [0, 255, 255], "label": "Bookshelf"},
    {"color": [255, 0, 255], "label": "Desk/Table/Counter"},
    {"color": [255, 255, 0], "label": "Chair"},
    {"color": [255, 0, 0], "label": "Book/Paper"},
    {"color": [0, 255, 0], "label": "Picture"},
    {"color": [0, 0, 255], "label": "Window"},
    {"color": [0, 0, 0], "label": "Door"}
]

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

    next_button_axis = plt.axes([0.7, 0.05, 0.1, 0.075])
    next_button = Button(next_button_axis, "Next")

    def next_plot(_):
        """
        Close the plot to show next plot.
        """
        plt.close()

    next_button.on_clicked(next_plot)

    exit_button_axis = plt.axes([0.85, 0.05, 0.1, 0.075])
    exit_button = Button(exit_button_axis, "Exit")

    def exit_plot(_):
        """
        Close the plot and exit.
        """
        plt.close()
        sys.exit(0)

    exit_button.on_clicked(exit_plot)

    legend_pathes = []
    for label_and_color in CLASS_LABELS_AND_COLORS:
        patch = patches.Patch(color=[c/255 for c in label_and_color["color"]], label=label_and_color["label"])
        legend_pathes.append(patch)
    plt.legend(handles=legend_pathes, bbox_to_anchor=(-2.0, 2.5))

    plt.show()

    #should_continue = input("Do you want to continue? [y/n] ")
    #if should_continue is "n":
    #    sys.exit(0)

def create_mask(pred_mask):
    """
    Get the mask from a prediction.
    """
    pred_mask = np.argmax(pred_mask, axis=3)
    print(pred_mask)
    print(np.shape(pred_mask))
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
        mask = np.zeros((224, 224, 3), dtype=np.int32)

        for w in range(0, 224):
            for h in range(0, 224):
                mask[h][w] = CLASS_LABELS_AND_COLORS[label[h][w]]["color"]

        pred_mask = model.predict(np.stack([image], axis=0))
        pred_mask = create_mask(pred_mask)

        pred_mask_new = np.zeros((224, 224, 3), dtype=np.int32)
        for h in range(0, 224):
            for w in range(0, 224):
                pred_mask_new[h][w] = CLASS_LABELS_AND_COLORS[pred_mask[h][w]]["color"]
        display([image, mask, pred_mask_new])

if __name__ == "__main__":
    sys.exit(main())
