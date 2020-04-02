"""
Calculate the class accuracy on the test set given a class ID.
"""

import sys
import os
import argparse
import json
import random
from PIL import Image
import numpy as np
import tensorflow as tf

from metrics import custom_balanced_accuracy, custom_accuracy, balanced_accuracy

from models.segnet import segnet_model

# Model
MODEL = segnet_model

DATA_MEAN = [0.491024, 0.455375, 0.427466]
DATA_STD = [0.262995, 0.267877, 0.270293]

# Number of classes per pixel
NUM_CLASSES = 8

# Width and height of input image
WIDTH = 224
HEIGHT = 224

def create_mask(pred_mask):
    """
    Get the mask from a prediction.
    """
    pred_mask = np.argmax(pred_mask, axis=3)
    return pred_mask[0]

def calculate_accuracy(model, data_ids, images_path, annotations_path, normalize):
    """
    Calculate the accuracy.
    """
    mean = np.array(DATA_MEAN)
    std = np.array(DATA_STD)

    metrics = {}
    for class_id in range(0, NUM_CLASSES):
        metrics[class_id] = balanced_accuracy()

    results = {}
    for class_id in range(0, NUM_CLASSES):
        results[class_id] = []

    for idx, data_id in enumerate(data_ids):
        print("{} of {}".format(idx + 1, len(data_ids)))

        image_file = Image.open(os.path.join(images_path, data_id + ".jpg"))
        image = np.asarray(image_file)

        annotation_file = open(os.path.join(annotations_path, data_id + ".json"), "r")
        annotation_data = json.load(annotation_file)
        label = np.array(annotation_data["annotation"], dtype=np.uint8)

        if normalize:
            image = (image - mean) / std

        pred_mask = model.predict(np.stack([image], axis=0))
        pred_mask = create_mask(pred_mask)

        for class_id in range(0, NUM_CLASSES):
            pred = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
            ground_truth_label = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

            for h in range(0, HEIGHT):
                for w in range(0, WIDTH):
                    ground_truth_label[h][w] = 1 if label[h][w] == class_id else 0

            for h in range(0, 224):
                for w in range(0, 224):
                    pred[h][w] = 1 if pred_mask[h][w] == class_id else 0

            metric = metrics[class_id](ground_truth_label, pred)
            result = results[class_id]
            if metric >= 0.0:
                result.append(metric)
            results[class_id] = result

            print("\tclass {}: {}".format(class_id, metric))

    for class_id in range(0, NUM_CLASSES):
        result = results[class_id]
        result = np.array(result, dtype=np.float32)

        print("Metric for class {}: {} with STD {}".format(class_id, np.mean(result), np.std(result)))

def main():
    """
    Model setup and training.
    """
    print(
        "#####################################\n"
        "# WARNING: May not work with TF GPU #\n"
        "#####################################\n\n"
    )

    parser = argparse.ArgumentParser(description="Calculate class ID.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("checkpoint_file", type=str, help="Checkpoint file.")
    parser.add_argument("--normalize", action="store_true", help="Normalize input.")
    args = parser.parse_args()

    with tf.device("/cpu:0"):
        model = MODEL(WIDTH, HEIGHT, NUM_CLASSES)

    if args.checkpoint_file:
        model.load_weights(args.checkpoint_file)

    images_path = os.path.join(args.data_path, "images")
    annotations_path = os.path.join(args.data_path, "annotations")

    train_test_data_split_file = open(os.path.join(args.data_path, "train_test_data_split.json"))
    train_test_data_split = json.load(train_test_data_split_file)

    test_set_ids = train_test_data_split["test"]

    calculate_accuracy(model, test_set_ids, images_path, annotations_path, args.normalize)

if __name__ == "__main__":
    sys.exit(main())
