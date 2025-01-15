import cv2
import mediapipe as mp

from arm_controls import *

arm = Arm_Controller()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            pointer_finger_tip = hand_landmarks.landmark[8]
            x, y, z = pointer_finger_tip.x, pointer_finger_tip.y, pointer_finger_tip.z

            #print(f'Index Finger Tip - X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}')
   
            target_position = [x - 0.5, 0.7 - y, z - 0.1]

            target_position[0] = round(target_position[0]*0.6, 2)
            target_position[1] = round(target_position[1]*0.6, 2)
            target_position[2] = round(target_position[2]*2.0, 2)
            
            print(f'Index Finger Tip - X: {target_position[0]:.4f}, Y: {target_position[1]:.4f}, Z: {target_position[2]:.4f}')
            arm.iterate_inverse_kinematics(target_position)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
