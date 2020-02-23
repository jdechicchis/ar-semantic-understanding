"""
Get Intel Realsense RGB-D data asynchronously.
"""

import pyrealsense2 as rs
import _thread

class CameraData():
    """
    Access color and depth for latest color and depth frames.
    """
    def __init__(self):
        self.color = None
        self.depth = None

    def start(self):
        """
        Start getting camera input.
        """
        def start_thread():
            pipeline = rs.pipeline()
            pipeline.start()

            try:
                while True:
                    frames = pipeline.wait_for_frames()

                    self.color = frames.get_color_frame()
                    self.depth = frames.get_depth_frame()
            finally:
                pipeline.stop()

        _thread.start_new_thread(start_thread, ())
