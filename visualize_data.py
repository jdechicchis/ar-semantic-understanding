"""
Visualize SUN RGB-D data.
"""

import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PLOT_COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow"]

class Visualizer:
    """
    Load image and annotation and display the RGB image and RGB image with segmentation overlay
    side-by-side.
    """
    def __init__(self, data_path):
        assert os.path.exists(data_path), "Data path provided ivalid."

        image_dir = os.path.join(data_path, "image")
        annotation_dir = os.path.join(data_path, "annotation2Dfinal")

        assert os.path.exists(image_dir), "Data path does not contain an image dir."
        assert os.path.exists(annotation_dir), "Data path does not cotain an annotation dir."

        def get_image_file_name():
            jpg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
            assert len(jpg_files) == 1, "Must be one image in image dir."
            return os.path.join(image_dir, jpg_files[0])

        self.__image_file = get_image_file_name()
        self.__annotation_file = os.path.join(annotation_dir, "index.json")

        assert os.path.isfile(self.__image_file), "Image file does not exist."
        assert os.path.isfile(self.__annotation_file), "Annotation file does not exist."

    def display_image(self):
        """
        Display the image with segmentation ground truth.
        """
        image = plt.imread(self.__image_file)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        annotations = self.__read_annotation_file()

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

            patch = patches.Polygon(annotation["coordinates"],
                                    facecolor=color,
                                    label=name,
                                    alpha=0.5)
            ax.add_patch(patch)

        handles, labels = ax.get_legend_handles_labels()

        plt.legend(handles=handles, ncol=2, bbox_to_anchor=(1.0, 1.0))

        plt.show()

    def __read_annotation_file(self):
        annotation_json = open(self.__annotation_file)
        annotations = json.load(annotation_json)

        objects = annotations["objects"]

        formatted_annotations = []

        for annotation in annotations["frames"][0]["polygon"]:
            if annotation["x"] and annotation["y"] and len(annotation["x"]) == len(annotation["y"]):
                coordinates = []
                for x, y in zip(annotation["x"], annotation["y"]):
                    coordinates.append((x, y))
                formatted_annotations.append({
                    "coordinates": coordinates,
                    "name": objects[annotation["object"]]["name"]
                })
            else:
                print("Ivalid annotation: %s" % self.__annotation_file)
        return formatted_annotations

def main():
    """
    Parse arguments and visualize the specified image.
    """
    parser = argparse.ArgumentParser(description="Visualize SUN RGB-D data.")
    parser.add_argument("data_path", type=str, help="Path to the data to visualize.")
    args = parser.parse_args()

    visualizer = Visualizer(args.data_path)
    visualizer.display_image()

if __name__ == "__main__":
    sys.exit(main())
