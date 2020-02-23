"""
Edge server.
"""

import argparse
import json
import time
from flask import Flask
import _thread

from utils.camera_data import CameraData
from utils.run_inference import RunInference, CLASS_LABELS_AND_COLORS
from utils.camera_visualizer import CameraVisualizer

APP = Flask(__name__)

CAMERA_DATA = CameraData()
RUN_INFERENCE = RunInference()
CAMERA_VISUALIZER = CameraVisualizer()

class InferenceResult():
    """
    Encapsulate inference result and image/depth data.
    """
    def __init__(self):
        self.color_image = None
        self.pred_mask = None
        self.depth_data = None

    def set_results(self, color_image, pred_mask, depth_data):
        """
        Set inference results.
        """
        self.color_image = color_image
        self.pred_mask = pred_mask
        self.depth_data = depth_data

    def get_results(self):
        """
        Get the inference results.
        """
        return self.color_image, self.pred_mask, self.depth_data

INFERENCE_RESULT = InferenceResult()

def visualize():
    """
    Visualize the inference result.
    """
    while True:
        color_image, pred_mask, depth_data = INFERENCE_RESULT.get_results()
        if not color_image is None and not pred_mask is None and not depth_data is None:
            CAMERA_VISUALIZER.visualize(color_image, depth_data, pred_mask, CLASS_LABELS_AND_COLORS)
        time.sleep(0.5)

def rgb_to_hex(r, g, b):
    """
    Convert RGB color to hex color.
    """
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

@APP.route('/')
def welcome_message():
    """
    Just return hellow world to test the server.
    """
    return "The server is up and running! Go to /mask to get JSON data."

@APP.route('/mask')
def mask():
    """
    Return the segmentation mask information.
    """
    color = CAMERA_DATA.color
    depth = CAMERA_DATA.depth

    color_image, depth_data, pred_mask, pred_mask_colors, _ = RUN_INFERENCE.run_inference(color, depth, False)

    INFERENCE_RESULT.set_results(color_image, pred_mask_colors, depth_data)

    locations = []
    for h in range(0, 224):
        for w in range(0, 224):
            color = pred_mask_colors[h][w]
            color = rgb_to_hex(color[0], color[1], color[2])
            if pred_mask[h][w] != 0:
                locations.append({"x": float(w), "y": float(h), "z": float(depth_data[h][w]), "color": color})

    return {"locations": locations}

def main():
    """
    Parse arguments and start server.
    """
    parser = argparse.ArgumentParser(description="Run edge server.")
    parser.add_argument("--public", action="store_true", help="Make server publically visible.")
    args = parser.parse_args()

    CAMERA_DATA.start()
    _thread.start_new_thread(visualize, ())

    if args.public:
        print("\n########## RUNNING SERVER PUBLICLY!!!! ##########\n")
        APP.run(host="0.0.0.0", port=5005)
    else:
        APP.run()

if __name__ == '__main__':
    main()
