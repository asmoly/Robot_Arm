import os
import pyzed.sl as sl
import cv2
import math
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import time
import pickle
import copy
import keyboard
import threading

from arm_controls import *

RECORDING_FPS = 10

CAMERA_DIM = (1280, 720)
FOCAL_LENGTH_HD720 = 736

pipeline = rs.pipeline()
pipeline.start()

current_episode = 0
episode_image_counter = 0

def data_writer_thread(depth_image, color_image, arm_angles, gripper_state, counter):
    arm_state = [arm_angles, gripper_state]
    cv2.imwrite(f"data/{current_episode}/depth_images/{counter}.png", depth_image)
    cv2.imwrite(f"data/{current_episode}/color_images/{counter}.png", color_image)
    pickle.dump(arm_state, open(f"data/{current_episode}/arm_positions/{counter}", "wb"))

# Center coordinates from x (0 - 1), y (1 - 0) to pixels centered in the center of the screen
def center_coord(x, y):
    x_converted = (x - 0.5)*CAMERA_DIM[0]
    y_converted = ((1 - y) - 0.5)*CAMERA_DIM[1]
    return int(x_converted), int(y_converted)

def main():
    global safe_to_read
    global safe_to_write
    global data_buffer
    global current_episode
    
    collecting_data = False

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

    old_time = time.time()

    # 1 - open, 0 - closed
    gripper_state = 1

    key = ' '

    print("Beggining main loop")
    while key != 113:
        if keyboard.is_pressed("a") and collecting_data == False:
            episode_image_counter = 0
            episode_count = pickle.load(open("data/data_count", "rb"))
            print(f"Current Episode Count: {episode_count}")
            print(f"Collecting Data for Episode {episode_count + 1}")
            current_episode = episode_count + 1

            os.makedirs(f"data/{current_episode}", exist_ok=True)
            os.makedirs(f"data/{current_episode}/depth_images", exist_ok=True)
            os.makedirs(f"data/{current_episode}/color_images", exist_ok=True)
            os.makedirs(f"data/{current_episode}/arm_positions", exist_ok=True)

            collecting_data = True
        elif keyboard.is_pressed("s"):
            collecting_data = False
            pickle.dump(current_episode, open("data/data_count", "wb"))
            print("Stopped Collecting Data")

        
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Zed Camera
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            image_ocv = image.get_data()
            depth_image_ocv = depth_image.get_data()
            depth_image_ocv = np.nan_to_num(depth_image_ocv)

            image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            rgb_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)

            # Realsense Camera
            frames_rs = pipeline.wait_for_frames()
            depth_rs = frames_rs.get_depth_frame()
            color_rs = frames_rs.get_color_frame()

            depth_data_rs = depth_rs.as_frame().get_data()
            depth_image_rs = np.asanyarray(depth_data_rs)
            depth_image_normalized_rs = (depth_image_rs/depth_image_rs.max())*255
            depth_image_normalized_rs = depth_image_normalized_rs.astype(np.uint8)

            color_data_rs = color_rs.as_frame().get_data()
            color_image_rs = np.asanyarray(color_data_rs)

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

                            gripper_angle = 150
                            gripper_state = 1
                            if pointer_to_thumb_distance <= 0.03:
                                gripper_angle = 200
                                gripper_state = 0
                                #print("Closed")

                            target_position = [wrist_pos_world[0], wrist_pos_world[1], -(2 - abs(wrist_pos_world[2]))]
                            arm.iterate_inverse_kinematics(target_position, gripper_angle=gripper_angle)
            else:
                #arm.reset_all_joints()
                pass

            # cv2.imshow("Arm view", image_ocv)
            # cv2.imshow("depth View", depth_image_rs)
            # cv2.imshow("Color View", color_image_rs)

            if (time.time() - old_time) >= 1.0/RECORDING_FPS and collecting_data == True:
                threading.Thread(target=data_writer_thread, args=(depth_image_rs, color_image_rs, arm.joint_angles, gripper_state, episode_image_counter)).start()
                episode_image_counter += 1

            height = 200
            image_ocv = cv2.resize(image_ocv, (int(image_ocv.shape[1]*height/image_ocv.shape[0]), height))
            depth_image_normalized_rs = cv2.resize(depth_image_normalized_rs, (int(depth_image_normalized_rs.shape[1]*height/depth_image_normalized_rs.shape[0]), height))
            color_image_rs = cv2.resize(color_image_rs, (int(color_image_rs.shape[1]*height/color_image_rs.shape[0]), height))

            depth_image_normalized_rs = cv2.cvtColor(depth_image_normalized_rs, cv2.COLOR_GRAY2BGR)

            canvas_height = max(image_ocv.shape[0], depth_image_normalized_rs.shape[0], color_image_rs.shape[0]) * 2
            canvas_width = max(image_ocv.shape[1], depth_image_normalized_rs.shape[1], color_image_rs.shape[1]) * 2
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Place images in the corners
            canvas[:image_ocv.shape[0], :image_ocv.shape[1]] = image_ocv  # Top-left corner
            canvas[:depth_image_normalized_rs.shape[0], -depth_image_normalized_rs.shape[1]:] = depth_image_normalized_rs  # Top-right corner
            canvas[-color_image_rs.shape[0]:, :color_image_rs.shape[1]] = color_image_rs  
            #cv2.imshow("main", np.concatenate((image_ocv, depth_image_rs, color_image_rs), axis=1))
            cv2.imshow("main", canvas)

        key = cv2.waitKey(1)

    zed.close()

if __name__ == "__main__":
    main()
