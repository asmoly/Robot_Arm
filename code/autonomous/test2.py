import cv2
import numpy as np

image = cv2.imread("data/test/190/depth_images/50.png", cv2.IMREAD_UNCHANGED)
print(np.median(image))
print(np.average(image))