"""
Run inference on camera input fom Intel Realsense.
"""

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pptk
from PIL import Image

from segnet import segnet_model

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

model = segnet_model(224, 224, 8)
model.load_weights("../model/checkpoints/experiment_segnet_normalized_ff615e2d5.weights.best.hdf5")

def create_mask(pred_mask):
    """
    Get the mask from a prediction.
    """
    pred_mask = np.argmax(pred_mask, axis=3)
    return pred_mask[0]

pipeline = rs.pipeline()
pipeline.start()
try:
    original_image = None
    segmented_image = None
    depth_image = None
    depth_color_overlay = None
    depth_viewer = None

    mean = np.array(DATA_MEAN)
    std = np.array(DATA_STD)

    for i in range(0, 1000):
        frames = pipeline.wait_for_frames()

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        depth_data = np.zeros((480, 640))

        for x in range(0, 640):
            for y in range(0, 480):
                dist = depth.get_distance(x, y)
                if dist > 0 and dist <= 2:
                    depth_data[y][x] = dist

        """
        depth_data_array = []
        for y in range(0, 480):
            for x in range(0, 640):
                if depth_data[y][x] > 0 and depth_data[y][x] <= 2:
                    # look towards green going away
                    depth_data_array.append([-x / 640, depth_data[y][x], -y / 480])
        if depth_data_array:
            depth_data_array = np.array(depth_data_array)
        else:
            depth_data_array = np.array([[0, 0, 1]])
        """

        depth = Image.fromarray(depth_data)
        depth = depth.resize((int(224 * (640.0 / 480.0)), 224), Image.NEAREST)

        post_resize_image_width, post_resize_image_height = depth.size

        new_width = 150
        left = (post_resize_image_width - new_width) / 2
        top = (post_resize_image_height - new_width) / 2
        right = (post_resize_image_width + new_width) / 2
        bottom = (post_resize_image_height + new_width) / 2

        left_point = 65
        top_point = 40
        depth = depth.crop((left_point, top_point, left_point + new_width, top_point + new_width))

        depth = depth.resize((224, 224), Image.NEAREST)

        depth_data = np.array(depth)

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
        normalized_image = (inference_image - mean) / std

        pred_mask = model.predict(np.stack([normalized_image], axis=0))

        pred_mask = create_mask(pred_mask)

        pred_mask_new = np.zeros((224, 224, 3), dtype=np.int32)
        for h in range(0, 224):
            for w in range(0, 224):
                pred_mask_new[h][w] = CLASS_LABELS_AND_COLORS[pred_mask[h][w]]["color"]

        if original_image and segmented_image and depth_image and depth_color_overlay:
            original_image.set_data(color_image)

            segmented_image.clear()
            segmented_image.imshow(color_image, alpha=1.0)
            segmented_image.imshow(pred_mask_new, alpha=0.5)

            depth_image.clear()
            depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=1)
            depth_image.imshow(pred_mask_new, alpha=0.5)

            depth_color_overlay.clear()
            depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=2)
            depth_color_overlay.imshow(color_image, alpha=0.3)

            #depth_viewer.clear()
            #depth_viewer.load(depth_data_array)
            #depth_viewer.set(point_size=0.001, lookat=[0, 0, 0], phi=1.6, theta=0.2, r=8)
        else:
            original_image = plt.subplot(2, 2, 1)
            original_image = original_image.imshow(color_image)

            segmented_image = plt.subplot(2, 2, 2)
            segmented_image.imshow(color_image, alpha=1.0)
            segmented_image.imshow(pred_mask_new, alpha=0.5)

            depth_image = plt.subplot(2, 2, 3)
            depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=1)
            depth_image.imshow(pred_mask_new, alpha=0.5)

            depth_color_overlay = plt.subplot(2, 2, 4)
            depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=2)
            depth_color_overlay.imshow(color_image, alpha=0.3)

            #depth_viewer = pptk.viewer(depth_data_array)
            #depth_viewer.set(point_size=0.001, lookat=[0, 0, 0], phi=1.5, theta=0.2, r=1)

        legend_pathes = []
        for label_and_color in CLASS_LABELS_AND_COLORS:
            patch = patches.Patch(color=[c/255 for c in label_and_color["color"]], label=label_and_color["label"])
            legend_pathes.append(patch)
        plt.legend(handles=legend_pathes, bbox_to_anchor=(0.25, -0.1))

        #if i == 2:
        #    break

        plt.pause(0.05)
        plt.draw()
    plt.close()

finally:
    pipeline.stop()
