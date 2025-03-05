import pyzed.sl as sl
import cv2
import math
import numpy as np
import mediapipe as mp

from arm_controls import *

CAMERA_DIM = (1280, 720)
FOCAL_LENGTH_HD720 = 736

# Center coordinates from x (0 - 1), y (1 - 0) to pixels centered in the center of the screen
def center_coord(x, y):
    x_converted = (x - 0.5)*CAMERA_DIM[0]
    y_converted = ((1 - y) - 0.5)*CAMERA_DIM[1]
    return int(x_converted), int(y_converted)

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
                    wrist_depth_coord_x, wrist_depth_coord_y = int(wrist.x*CAMERA_DIM[0]), int(wrist.y*CAMERA_DIM[1])
                    wrist_x, wrist_y = center_coord(wrist.x, wrist.y)

                    pointer_finger = hand_landmarks.landmark[8]
                    pointer_x, pointer_y = center_coord(pointer_finger.x, pointer_finger.y)

                    thumb_finger = hand_landmarks.landmark[4]
                    thumb_x, thumb_y = center_coord(thumb_finger.x, thumb_finger.y)

                    if wrist_depth_coord_x >= 0 and wrist_depth_coord_x < CAMERA_DIM[0] and wrist_depth_coord_y >= 0 and wrist_depth_coord_y < CAMERA_DIM[1]:
                        depth = depth_image_ocv[wrist_depth_coord_y][wrist_depth_coord_x]

                        if depth != 0:
                            # mm to m
                            depth = (depth/1000)

                            wrist_pos_world = [depth*wrist_x/FOCAL_LENGTH_HD720, depth*wrist_y/FOCAL_LENGTH_HD720, -depth]
                            pointer_pos_world = [depth*pointer_x/FOCAL_LENGTH_HD720, depth*pointer_y/FOCAL_LENGTH_HD720, -depth]
                            thumb_pos_world = [depth*thumb_x/FOCAL_LENGTH_HD720, depth*thumb_y/FOCAL_LENGTH_HD720, -depth]

                            # Distance from pointer finger to thumb in meters
                            pointer_to_thumb_distance = math.sqrt((pointer_pos_world[0] - thumb_pos_world[0])**2 + (pointer_pos_world[1] - thumb_pos_world[1])**2)
                            #print(pointer_to_thumb_distance)

                            gripper_angle = 150
                            if pointer_to_thumb_distance <= 0.03:
                                gripper_angle = 200
                                print("Closed")

                            target_position = [wrist_pos_world[0], wrist_pos_world[1], -(1.8 - abs(wrist_pos_world[2]))]
                            arm.iterate_inverse_kinematics(target_position, gripper_angle=gripper_angle)
                            print(arm.joint_angles)
            else:
                #arm.reset_all_joints()
                pass

            cv2.imshow("Normal View", image_ocv)

        key = cv2.waitKey(1)

    zed.close()

if __name__ == "__main__":
    main()
