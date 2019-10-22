"""
Visualize SUN RGB-D data.
"""

import sys
import os
import argparse

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
        pass

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
