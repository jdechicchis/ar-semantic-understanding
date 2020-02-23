"""
Run inference on camera input fom Intel Realsense.
"""

import sys
import argparse

from utils.camera_data import CameraData
from utils.run_inference import RunInference, CLASS_LABELS_AND_COLORS
from utils.camera_visualizer import CameraVisualizer

def main():
    """
    Run inference on RGB-D data and visualize.
    """
    parser = argparse.ArgumentParser(description="Run inference on RGB-D data and visualize.")
    parser.add_argument("--plot_point_cloud", action="store_true", help="Plot point cloud.")
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

        color_image, depth_data, _, pred_mask_colors, depth_data_array = \
            run_inference.run_inference(color, depth, args.plot_point_cloud)

        if args.plot_point_cloud:
            camera_visualizer.visualize(color_image,
                                        depth_data,
                                        pred_mask_colors,
                                        CLASS_LABELS_AND_COLORS,
                                        depth_data_array)
        else:
            camera_visualizer.visualize(color_image, depth_data, pred_mask_colors, CLASS_LABELS_AND_COLORS)

if __name__ == "__main__":
    sys.exit(main())
