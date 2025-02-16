import pyrealsense2 as rs
import numpy as np
import cv2

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)
        depth_image_normalized = (depth_image/depth_image.max())

        color_data = color.as_frame().get_data()
        color_image = np.asanyarray(color_data)

        cv2.imshow("video", color_image)
        cv2.imshow("depth", depth_image_normalized)
        cv2.waitKey(1)

finally:
    pipeline.stop()