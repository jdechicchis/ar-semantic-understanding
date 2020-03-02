"""
Visualize camera data and inference result for Intel Realsense.
"""

import os
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np

class CameraVisualizer():
    """
    Visualize camera data and inference result.
    """
    def __init__(self):
        self.original_image = None
        self.segmented_image = None
        self.depth_image = None
        self.depth_color_overlay = None
        self.depth_viewer = None

    def visualize(self,
                  color_image,
                  depth_data,
                  pred_mask_color,
                  class_label_colors,
                  pred_mask,
                  depth_data_array,
                  save_dir):
        """
        Visualize inference result.
        """
        if self.original_image and self.segmented_image and self.depth_image and self.depth_color_overlay:
            self.original_image.set_data(color_image)

            self.segmented_image.clear()
            self.segmented_image.imshow(color_image, alpha=1.0)
            self.segmented_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_image.clear()
            self.depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=5.0)
            self.depth_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_color_overlay.clear()
            self.depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=5.0)
            self.depth_color_overlay.imshow(color_image, alpha=0.3)
        else:
            self.original_image = plt.subplot(2, 2, 1)
            self.original_image = self.original_image.imshow(color_image)

            self.segmented_image = plt.subplot(2, 2, 2)
            self.segmented_image.imshow(color_image, alpha=1.0)
            self.segmented_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_image = plt.subplot(2, 2, 3)
            self.depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=5.0)
            self.depth_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_color_overlay = plt.subplot(2, 2, 4)
            self.depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=5.0)
            self.depth_color_overlay.imshow(color_image, alpha=0.3)

        legend_pathes = []
        for label_and_color in class_label_colors:
            patch = patches.Patch(color=[c/255 for c in label_and_color["color"]],
                                  label=label_and_color["label"])
            legend_pathes.append(patch)
        plt.legend(handles=legend_pathes, bbox_to_anchor=(0.25, -0.1))

        save_button_axis = plt.axes([0.1, 0.05, 0.2, 0.075])
        save_button = Button(save_button_axis, "Save Point Cloud")

        def save_point_cloud(_):
            """
            Save the current point cloud.
            """
            current_time = datetime.datetime.utcnow()
            current_time_file_prefix = current_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
            file_name = os.path.join(save_dir, current_time_file_prefix) + ".json"

            print("Save point cloud: {}".format(file_name))

            # Index mask as [h][w] for the class of
            # each pixel
            mask_array = pred_mask.tolist()

            # Depth is stored as array of [x, y, z]
            # where x is in range [0, 640] and y is
            # in range [0, 480] and z is a float in
            # meters
            depth_array = depth_data_array.tolist()
            data = {
                "mask": mask_array,
                "depth": depth_array
            }

            with open(file_name, "w+") as outfile:
                json.dump(data, outfile)

        save_button.on_clicked(save_point_cloud)

        plt.draw()
        plt.pause(0.05)
