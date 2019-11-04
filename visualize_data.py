"""
Visualize SUN RGB-D data.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PLOT_COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow"]

OBJECT_COLORS = {
    "airconditioner": [1.0, 1.0, 1.0],
    "airduct": [1.0, 1.0, 1.0],
    "air_vent": [1.0, 1.0, 1.0],
    "bag": [1.0, 1.0, 1.0],
    "ball": [1.0, 1.0, 1.0],
    "basket": [1.0, 1.0, 1.0],
    "books": [1.0, 1.0, 1.0],
    "bookshelf": [1.0, 1.0, 1.0],
    "bottle": [1.0, 1.0, 1.0],
    "bowl": [1.0, 1.0, 1.0],
    "box": [1.0, 1.0, 1.0],
    "blinds": [1.0, 1.0, 1.0],
    "bulletin_board": [1.0, 1.0, 1.0],
    "cabinet": [1.0, 1.0, 1.0],
    "cablemodem": [1.0, 1.0, 1.0],
    "cable_modem": [1.0, 1.0, 1.0],
    "cablerack": [1.0, 1.0, 1.0],
    "camera": [1.0, 1.0, 1.0],
    "ceiling": [1.0, 1.0, 1.0],
    "chair": [1.0, 1.0, 1.0],
    "clipboard": [1.0, 1.0, 1.0],
    "clock": [1.0, 1.0, 1.0],
    "column": [1.0, 1.0, 1.0],
    "computer": [1.0, 1.0, 1.0],
    "cone": [1.0, 1.0, 1.0],
    "counter": [1.0, 1.0, 1.0],
    "cup": [1.0, 1.0, 1.0],
    "curtain": [1.0, 1.0, 1.0],
    "desk": [1.0, 1.0, 1.0],
    "doll": [1.0, 1.0, 1.0],
    "door": [1.0, 1.0, 1.0],
    "dishwasher": [1.0, 1.0, 1.0],
    "divider": [1.0, 1.0, 1.0],
    "electrical_outlet": [1.0, 1.0, 1.0],
    "envelope": [1.0, 1.0, 1.0],
    "eraser": [1.0, 1.0, 1.0],
    "exit_sign": [1.0, 1.0, 1.0],
    "flowers": [1.0, 1.0, 1.0],
    "garbage_bin": [1.0, 1.0, 1.0],
    "handsanitizer": [1.0, 1.0, 1.0],
    "hanginghooks": [1.0, 1.0, 1.0],
    "heater": [1.0, 1.0, 1.0],
    "hole_puncher": [1.0, 1.0, 1.0],
    "fan": [1.0, 1.0, 1.0],
    "faucet": [1.0, 1.0, 1.0],
    "fax_machine": [1.0, 1.0, 1.0],
    "file": [1.0, 1.0, 1.0],
    "fire_place": [1.0, 1.0, 1.0],
    "floor": [1.0, 1.0, 1.0],
    "folder": [1.0, 1.0, 1.0],
    "fridge": [1.0, 1.0, 1.0],
    "hook": [1.0, 1.0, 1.0],
    "jar": [1.0, 1.0, 1.0],
    "keyboard": [1.0, 1.0, 1.0],
    "ladder": [1.0, 1.0, 1.0],
    "laptop": [1.0, 1.0, 1.0],
    "light": [1.0, 1.0, 1.0],
    "magazine": [1.0, 1.0, 1.0],
    "magnet": [1.0, 1.0, 1.0],
    "mantle": [1.0, 1.0, 1.0],
    "map": [1.0, 1.0, 1.0],
    "microwave": [1.0, 1.0, 1.0],
    "monitor": [1.0, 1.0, 1.0],
    "mouse": [1.0, 1.0, 1.0],
    "paper": [1.0, 1.0, 1.0],
    "paper_cutter": [1.0, 1.0, 1.0],
    "papertowel": [1.0, 1.0, 1.0],
    "paperrack": [1.0, 1.0, 1.0],
    "papertoweldispenser": [1.0, 1.0, 1.0],
    "pen": [1.0, 1.0, 1.0],
    "pencil_holder": [1.0, 1.0, 1.0],
    "picture": [1.0, 1.0, 1.0],
    "pipe": [1.0, 1.0, 1.0],
    "plant": [1.0, 1.0, 1.0],
    "pot": [1.0, 1.0, 1.0],
    "printer": [1.0, 1.0, 1.0],
    "projector": [1.0, 1.0, 1.0],
    "projector_screen": [1.0, 1.0, 1.0],
    "ruler": [1.0, 1.0, 1.0],
    "scissor": [1.0, 1.0, 1.0],
    "screen": [1.0, 1.0, 1.0],
    "shelf": [1.0, 1.0, 1.0],
    "shelves": [1.0, 1.0, 1.0],
    "sink": [1.0, 1.0, 1.0],
    "sofa": [1.0, 1.0, 1.0],
    "speaker": [1.0, 1.0, 1.0],
    "stamp": [1.0, 1.0, 1.0],
    "stand": [1.0, 1.0, 1.0],
    "stapler": [1.0, 1.0, 1.0],
    "stereo": [1.0, 1.0, 1.0],
    "stoveburner": [1.0, 1.0, 1.0],
    "styrofoamobject": [1.0, 1.0, 1.0],
    "table": [1.0, 1.0, 1.0],
    "tape": [1.0, 1.0, 1.0],
    "tape_dispenser": [1.0, 1.0, 1.0],
    "telephone": [1.0, 1.0, 1.0],
    "telephonecord": [1.0, 1.0, 1.0],
    "thermostat": [1.0, 1.0, 1.0],
    "vase": [1.0, 1.0, 1.0],
    "wall": [1.0, 1.0, 1.0],
    "watercarboy": [1.0, 1.0, 1.0],
    "water_purifier": [1.0, 1.0, 1.0],
    "window": [1.0, 1.0, 1.0],
    "wire": [1.0, 1.0, 1.0],
    "whiteboard": [1.0, 1.0, 1.0],
    "unknown": [0.0, 0.0, 0.0],
}

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
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        #fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        annotations = self.__read_annotation_file()

        print(np.shape(image))
        image_height = np.shape(image)[0]
        image_width = np.shape(image)[1]

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

            patch = patches.Polygon(annotation["coordinates"],
                                    facecolor=color,
                                    label=name,
                                    alpha=0.5)
            ax.add_patch(patch)

            x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            segment_path = patches.Polygon(annotation["coordinates"])
            grid = segment_path.contains_points(points, radius=1e-9)
            mask = grid.reshape(image_height, image_width)

            #TODO: Use mask to generate segment class for each pixel

            for h in range(0, image_height):
                for w in range(0, image_width):
                    if mask[h][w]:
                        segmented_image[h][w] = OBJECT_COLORS[annotation["name"]]

        handles, labels = ax.get_legend_handles_labels()

        plt.legend(handles=handles, ncol=2, bbox_to_anchor=(1.0, 1.0))

        ax = plt.subplot(1, 2, 2)
        ax.imshow(segmented_image)

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
