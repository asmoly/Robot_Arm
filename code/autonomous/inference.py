import pyrealsense2 as rs
import keyboard

from train import *
from arm_models import *
from arm_controls import *

PATH_TO_MODEL = "logs/run_20250301_231222/armnet_43.pt"#"logs/run_20250301_125238/armnet_20.pt"
# "logs/run_20250301_231222/armnet_43.pt"

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def main():
    arm = Arm_Controller()
    arm.reset_all_joints()

    pipeline = rs.pipeline()
    pipeline.start()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    arm_transformer_model = load_model(PATH_TO_MODEL).to(device)

    while keyboard.is_pressed("s") != True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)

        color_data = color.as_frame().get_data()
        color_image = np.asanyarray(color_data)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image = cv2.resize(color_image, (int(color_image.shape[1]/2), int(color_image.shape[0]/2)))
        depth_image = cv2.resize(depth_image, (int(depth_image.shape[1]/2), int(depth_image.shape[0]/2)))
        depth_image = np.clip(depth_image, None, REALSENSE_INF)

        frame = np.zeros((color_image.shape[0], color_image.shape[1], 4))
        frame[:, :, :3] = color_image/255.0
        frame[:, :, 3] = depth_image/REALSENSE_INF
        frame = np.array(frame, dtype=np.float32)

        frame = torch.as_tensor(frame)
        frame = torch.permute(frame, (2, 0, 1))

        arm_pose = torch.Tensor([arm.joint_angles[0], arm.joint_angles[1], arm.joint_angles[2], arm.gripper_state])
        print("Input:", arm_pose)
        arm_pose[:3] = arm_pose[:3]/180.0

        with torch.no_grad():
            frame = frame[None, :].to(device)
            arm_pose = arm_pose[None, :].to(device)
            output = arm_transformer_model(frame, arm_pose)
            
            output = output.cpu()[0]
            output[:3] = output[:3]*180
            output = output.int()

            print("Output:", output)
            output[0] = clamp(output[0], -45, 45)
            output[1] = clamp(output[1], -20, 80)
            output[2] = clamp(output[2], 0, 170)

            arm.rotate_all_joints(output[0], output[1], output[2], int(output[3]))

        cv2.imshow("video", color_image)
        cv2.imshow("depth", depth_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()