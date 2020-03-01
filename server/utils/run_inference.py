"""
Process and run inference on Intel Realsense RGB-D frame.
"""

import numpy as np
from PIL import Image
from .segnet import segnet_model

DATA_MEAN = [0.491024, 0.455375, 0.427466]
DATA_STD = [0.262995, 0.267877, 0.270293]

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

class RunInference():
    """
    Run inference on RGB-D data from Intel Realsense.
    """
    def __init__(self):
        self.mean = np.array(DATA_MEAN)
        self.std = np.array(DATA_STD)

        self.model = segnet_model(224, 224, 8)
        self.model.load_weights("../model/checkpoints/experiment_segnet_normalized_ff615e2d5.weights.best.hdf5")

    def run_inference(self, color, depth, point_cloud_array):
        """
        Run inference given color and depth frame.
        """
        # Get the depth data
        depth_data = np.zeros((480, 640))

        for x in range(0, 640):
            for y in range(0, 480):
                dist = depth.get_distance(x, y)
                if dist > 0 and dist <= 5.0:
                    depth_data[y][x] = dist

        depth_data_array = []
        if point_cloud_array:
            depth_data_array = []
            for y in range(0, 480):
                for x in range(0, 640):
                    if depth_data[y][x] > 0 and depth_data[y][x] <= 2:
                        depth_data_array.append([-x / 640, depth_data[y][x], -y / 480])
            if depth_data_array:
                depth_data_array = np.array(depth_data_array)
            else:
                depth_data_array = np.array([[0, 0, 1]])

        depth = Image.fromarray(depth_data)
        depth = depth.resize((int(224 * (640.0 / 480.0)), 224), Image.NEAREST)

        post_resize_image_width, post_resize_image_height = depth.size

        new_width = 150
        left = (post_resize_image_width - new_width) / 2
        top = (post_resize_image_height - new_width) / 2
        right = (post_resize_image_width + new_width) / 2
        bottom = (post_resize_image_height + new_width) / 2

        left_point = 67
        top_point = 40
        depth = depth.crop((left_point, top_point, left_point + new_width, top_point + new_width))

        depth = depth.resize((224, 224), Image.NEAREST)

        depth_data = np.array(depth)

        # Get the RGB data
        color_data = color.as_frame().get_data()
        color_image = np.asanyarray(color_data)
        color_image = Image.fromarray(color_image)

        color_image = color_image.resize((int(224 * (640.0 / 480.0)), 224))

        post_resize_image_width, post_resize_image_height = color_image.size

        left = (post_resize_image_width - 224) / 2
        top = (post_resize_image_height - 224) / 2
        right = (post_resize_image_width + 224) / 2
        bottom = (post_resize_image_height + 224) / 2

        color_image = color_image.crop((left, top, right, bottom))

        color_image = np.array(color_image)

        # Run inference
        inference_image = color_image / 255.0
        normalized_image = (inference_image - self.mean) / self.std

        pred_mask = self.model.predict(np.stack([normalized_image], axis=0))

        pred_mask = self.create_mask(pred_mask)

        pred_mask_new = np.zeros((224, 224, 3), dtype=np.int32)
        for h in range(0, 224):
            for w in range(0, 224):
                pred_mask_new[h][w] = CLASS_LABELS_AND_COLORS[pred_mask[h][w]]["color"]

        return color_image, depth_data, pred_mask, pred_mask_new, depth_data_array

    def create_mask(self, pred_mask):
        """
        Get the mask from a prediction.
        """
        pred_mask = np.argmax(pred_mask, axis=3)
        return pred_mask[0]
