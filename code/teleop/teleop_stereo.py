import pyzed.sl as sl
import cv2
import numpy as np
import mediapipe as mp

from arm_controls import *

CAMERA_DIM = (1280, 720)

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode

    arm = Arm_Controller()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera")
        exit(1)

    image = sl.Mat()
    depth_image = sl.Mat()

    key = ' '
    while key != 113:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            image_ocv = image.get_data()
            depth_image_ocv = depth_image.get_data()
            depth_image_ocv = np.nan_to_num(depth_image_ocv)


            image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            rgb_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)

            results = hands.process(rgb_image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_ocv, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                    wrist = hand_landmarks.landmark[0]
                    x, y = wrist.x, wrist.y

                    if x >= 0 and x < 1 and y >= 0 and y < 1:
                        depth = depth_image_ocv[int((y)*CAMERA_DIM[1])][int((x)*CAMERA_DIM[0])]

                        if depth != 0:
                            # mm to m
                            depth = 0.7 - (depth/10000)*5

                            target_position = [x - 0.5, 0.7 - y, -depth]

                            target_position[0] = round(target_position[0]*0.6, 2)
                            target_position[1] = round(target_position[1]*0.6, 2)
                            target_position[2] = round(target_position[2]*1.0, 2)
                            
                            print(f'Index Finger Tip - X: {target_position[0]:.4f}, Y: {target_position[1]:.4f}, Z: {target_position[2]:.4f}')
                            arm.iterate_inverse_kinematics(target_position)

            cv2.imshow("Normal View", image_ocv)

        key = cv2.waitKey(1)

    zed.close()

if __name__ == "__main__":
    main()
