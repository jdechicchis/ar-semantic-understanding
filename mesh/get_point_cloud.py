"""
Run inference on camera input fom Intel Realsense and choose
frame for which to save point cloud.
"""

import sys
import argparse

from utils.camera_data import CameraData
from utils.run_inference import RunInference, CLASS_LABELS_AND_COLORS
from utils.camera_visualizer import CameraVisualizer

def main():
    """
    Run inference on RGB-D data to visualize and save point cloud.
    """
    parser = argparse.ArgumentParser(description="Get points clouds and save them.")
    parser.add_argument("save_dir", help="Save destination dir.")
    args = parser.parse_args()

    camera_data = CameraData()
    run_inference = RunInference()
    camera_visualizer = CameraVisualizer()

    camera_data.start()

    while True:
        color = camera_data.color
        depth = camera_data.depth

        if not color or not depth:
            continue

        color_image, depth_data, pred_mask, pred_mask_colors, depth_data_array = \
            run_inference.run_inference(color, depth, True)

        camera_visualizer.visualize(color_image,
                                    depth_data,
                                    pred_mask_colors,
                                    CLASS_LABELS_AND_COLORS,
                                    pred_mask,
                                    depth_data_array,
                                    args.save_dir)

if __name__ == "__main__":
    sys.exit(main())
