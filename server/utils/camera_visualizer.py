"""
Visualize camera data and inference result for Intel Realsense.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pptk

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

    def visualize(self, color_image, depth_data, pred_mask_color, class_label_colors, depth_data_array=None):
        """
        Visualize inference result.
        """
        if self.original_image and self.segmented_image and self.depth_image and self.depth_color_overlay:
            self.original_image.set_data(color_image)

            self.segmented_image.clear()
            self.segmented_image.imshow(color_image, alpha=1.0)
            self.segmented_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_image.clear()
            self.depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=1)
            self.depth_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_color_overlay.clear()
            self.depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=2)
            self.depth_color_overlay.imshow(color_image, alpha=0.3)
        else:
            self.original_image = plt.subplot(2, 2, 1)
            self.original_image = self.original_image.imshow(color_image)

            self.segmented_image = plt.subplot(2, 2, 2)
            self.segmented_image.imshow(color_image, alpha=1.0)
            self.segmented_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_image = plt.subplot(2, 2, 3)
            self.depth_image.imshow(depth_data, cmap="gray", vmin=0, vmax=1)
            self.depth_image.imshow(pred_mask_color, alpha=0.5)

            self.depth_color_overlay = plt.subplot(2, 2, 4)
            self.depth_color_overlay.imshow(depth_data, cmap="gray", vmin=0, vmax=2)
            self.depth_color_overlay.imshow(color_image, alpha=0.3)

        legend_pathes = []
        for label_and_color in class_label_colors:
            patch = patches.Patch(color=[c/255 for c in label_and_color["color"]],
                                  label=label_and_color["label"])
            legend_pathes.append(patch)
        plt.legend(handles=legend_pathes, bbox_to_anchor=(0.25, -0.1))

        plt.draw()
        plt.pause(0.05)

        if not depth_data_array is None:
            if self.depth_viewer:
                self.depth_viewer.clear()
                self.depth_viewer.load(depth_data_array)
                self.depth_viewer.set(point_size=0.001, lookat=[0, 0, 0], phi=1.6, theta=0.2, r=8)
            else:
                self.depth_viewer = pptk.viewer(depth_data_array)
                self.depth_viewer.set(point_size=0.001, lookat=[0, 0, 0], phi=1.5, theta=0.2, r=1)
