"""
Preprocess SUN RGB-D data to have uniform image numbering and annotate each
pixel individually. Support image resizing as well.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PLOT_COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow"]
NAME_TO_RGB = {
    "blue": [0.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0],
    "red": [1.0, 0.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0]
}
TARGET_IMAGE_WIDTH_HEIGHT = 224

class Preprocessor():
    """
    Preprocess SUN RBG-D data.
    """
    def __init__(self, src_dir, src_dir_struct_file, dst_dir):
        assert os.path.exists(src_dir), "Src path provided ivalid."
        assert os.path.isfile(src_dir_struct_file), "Src dir struct file does not exist."
        assert os.path.exists(dst_dir), "Dst path provided ivalid."

        src_dir_struct = json.load(open(src_dir_struct_file, "r"))

        self.__src_dir = src_dir
        self.__src_dir_struct = src_dir_struct
        self.__dst_dir = dst_dir

    def preprocess(self):
        """
        Preprocess the data.
        """
        count = 0
        total_invalid = 0
        for data_dir in self.__src_dir_struct:
            data_dir = os.path.join(self.__src_dir, data_dir)
            #f for f in os.listdir(path) if not f.startswith('.')
            for subdir in os.listdir(data_dir):
                if not subdir.startswith("."):
                    num_invalid = self.__process_data(os.path.join(data_dir, subdir))
                    count += 1
                    total_invalid += num_invalid
        print("%d invalid annotations!" % total_invalid)
        print("Total data samples: %d" % count)

    def __process_data(self, data_dir):
        print("Processing: %s" % data_dir)
        image_dir = os.path.join(data_dir, "image")
        annotation_dir = os.path.join(data_dir, "annotation2Dfinal")

        assert os.path.exists(image_dir), "Data path does not contain an image dir."
        assert os.path.exists(annotation_dir), "Data path does not cotain an annotation dir."

        def get_image_file_name():
            jpg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
            assert len(jpg_files) == 1, "Must be one image in image dir."
            return os.path.join(image_dir, jpg_files[0])

        image_file = get_image_file_name()
        annotation_file = os.path.join(annotation_dir, "index.json")

        assert os.path.isfile(image_file), "Image file does not exist."
        assert os.path.isfile(annotation_file), "Annotation file does not exist."

        annotations, num_invalid = self. __read_annotation_file(annotation_file)

        image = Image.open(image_file)
        original_image_width, original_image_height = image.size
        width_height_ratio = original_image_width / original_image_height

        assert original_image_width >= original_image_height, "Image height is greater than width"

        image = image.resize((int(TARGET_IMAGE_WIDTH_HEIGHT * width_height_ratio), TARGET_IMAGE_WIDTH_HEIGHT))

        post_resize_image_width, post_resize_image_height = image.size

        left = (post_resize_image_width - TARGET_IMAGE_WIDTH_HEIGHT) / 2
        top = (post_resize_image_height - TARGET_IMAGE_WIDTH_HEIGHT) / 2
        right = (post_resize_image_width + TARGET_IMAGE_WIDTH_HEIGHT) / 2
        bottom = (post_resize_image_height + TARGET_IMAGE_WIDTH_HEIGHT) / 2

        image = image.crop((left, top, right, bottom))

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax.imshow(image)
        ax.axis("off")

        image_width, image_height = image.size
        segmented_image = np.zeros(np.shape(image))

        color_idx = 0
        color_assignments = {}
        for annotation in annotations:
            name = annotation["name"]

            if name in color_assignments:
                color = color_assignments[name]
            else:
                color = PLOT_COLORS[color_idx % (len(PLOT_COLORS) - 1)]
                color_idx += 1
                color_assignments[name] = color

            resized_coordinates = self.__resize_annotation_coordinates(annotation["coordinates"],
                                                                       original_image_width,
                                                                       original_image_height,
                                                                       post_resize_image_width,
                                                                       post_resize_image_height)

            cropped_coordinates = self.__crop_annotation_coordinates(resized_coordinates,
                                                                     post_resize_image_width,
                                                                     post_resize_image_height,
                                                                     image_width,
                                                                     image_height)

            patch = patches.Polygon(cropped_coordinates,
                                    facecolor=color,
                                    label=name,
                                    alpha=0.5)
            ax.add_patch(patch)

            x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            segment_path = patches.Polygon(cropped_coordinates)
            grid = segment_path.contains_points(points, radius=1e-9)
            mask = grid.reshape(image_height, image_width)

            for h in range(0, image_height):
                for w in range(0, image_width):
                    if mask[h][w]:
                        segmented_image[h][w] = NAME_TO_RGB[color]

        handles, _ = ax.get_legend_handles_labels()

        plt.legend(handles=handles, ncol=2, bbox_to_anchor=(1.0, 1.0))

        ax = plt.subplot(1, 2, 2)
        ax.imshow(segmented_image)

        plt.show()

        should_continue = input("Do you want to continue? [y/n] ")
        if should_continue is "n":
            sys.exit(0)

        return num_invalid

    def __resize_annotation_coordinates(self, coordinates, original_width, original_height, new_width, new_height):
        new_coordinates = []
        for coordinate in coordinates:
            new_x = coordinate[0] * (new_width / original_width)
            new_y = coordinate[1] * (new_height / original_height)
            new_coodinate = (new_x, new_y)
            new_coordinates.append(new_coodinate)
        return new_coordinates

    def __crop_annotation_coordinates(self, coordinates, original_width, original_height, new_width, new_height):
        x_change_amount = (original_width - new_width) / 2
        y_change_amount = (original_height - new_height) / 2
        new_coordinates = []
        for coordinate in coordinates:
            new_x = coordinate[0] - x_change_amount
            new_y = coordinate[1] - y_change_amount
            new_coodinate = (new_x, new_y)
            new_coordinates.append(new_coodinate)
        return new_coordinates

    def __read_annotation_file(self, annotation_file):
        num_invalid = 0
        annotation_json = open(annotation_file)
        annotations = json.load(annotation_json)

        objects = annotations["objects"]

        formatted_annotations = []

        for annotation in annotations["frames"][0]["polygon"]:
            if not isinstance(annotation["x"], list) or not isinstance(annotation["y"], list):
                print("Annotation contains non-list for x or y: %s" % annotation_file)
                num_invalid += 1
            elif annotation["object"] >= len(objects):
                print("Annotation object invalid: %s" % annotation_file)
                num_invalid += 1
            elif annotation["x"] and annotation["y"] and len(annotation["x"]) == len(annotation["y"]):
                coordinates = []
                for x, y in zip(annotation["x"], annotation["y"]):
                    coordinates.append((x, y))
                formatted_annotations.append({
                    "coordinates": coordinates,
                    "name": objects[annotation["object"]]["name"]
                })
            else:
                print("Ivalid annotation: %s" % annotation_file)
                num_invalid += 1
        return formatted_annotations, num_invalid

def main():
    """
    Parse arguments and preprocess the SUN RGB-D data.
    """
    parser = argparse.ArgumentParser(description="Preprocess SUN RGB-D data.")
    parser.add_argument("src_path", type=str, help="Path to the SUN RGB-D data to proprocess.")
    parser.add_argument("src_struct_file", type=str,
                        help="A json file which contains the dir structure.")
    parser.add_argument("dst_path", type=str, help="Path to store the preprocessed data.")
    args = parser.parse_args()

    preprocessor = Preprocessor(args.src_path, args.src_struct_file, args.dst_path)
    preprocessor.preprocess()

if __name__ == "__main__":
    sys.exit(main())
