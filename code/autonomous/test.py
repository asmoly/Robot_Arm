from torch.utils.data import DataLoader, Dataset
import torch
import os
import cv2
import numpy as np
import pickle
from math import cos, sin, pi
from matrix import *

class ArmEpisodeDataset(Dataset):
    def __init__(self, path_to_data, image_transform=None):
        self.path_to_data = path_to_data
        self.image_transform = image_transform

        frame_counter = 0
        self.frames = []
        for episode in os.listdir(self.path_to_data):
            full_path = os.path.join(self.path_to_data, episode)
            
            episode_frame_count = 0
            for root, dirs, files in os.walk(os.path.join(full_path, "color_images")):
                episode_frame_count += len(files)

            for frame in os.listdir(os.path.join(full_path, "color_images")):
                if int(frame.replace(".png", "")) != (episode_frame_count - 1):
                    self.frames.append([full_path, int(frame.replace(".png", ""))])

        self.data_length = len(self.frames)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        path_to_color_image = os.path.join(frame_path[0], "color_images", f"{frame_path[1]}.png")
        path_to_depth_image = os.path.join(frame_path[0], "depth_images", f"{frame_path[1]}.png")
        path_to_current_pose = os.path.join(frame_path[0], "arm_positions", f"{frame_path[1]}")
        path_to_future_pose = os.path.join(frame_path[0], "arm_positions", f"{frame_path[1] + 1}")

        color_image = cv2.imread(path_to_color_image)
        depth_image = cv2.imread(path_to_depth_image, cv2.IMREAD_GRAYSCALE)
        current_pose = pickle.load(open(path_to_current_pose, "rb"))
        future_pose = pickle.load(open(path_to_future_pose, "rb"))

        color_image = cv2.resize(color_image, (int(color_image.shape[1]/2), int(color_image.shape[0]/2)))
        depth_image = cv2.resize(depth_image, (int(depth_image.shape[1]/2), int(depth_image.shape[0]/2)))
                
        color_image = torch.Tensor(color_image)
        if self.image_transform != None:
            color_image = self.image_transform(torch.Tensor(color_image))

        depth_image = torch.Tensor(depth_image)

        image = torch.zeros((color_image.shape[0], color_image.shape[1], 4))
        # print(color_image.shape)
        # print(depth_image.shape)
        # print(image.shape)
        image[:, :, :3] = color_image
        image[:, :, 3] = depth_image
        
        current_pose = torch.Tensor([current_pose[0][0], current_pose[0][1], current_pose[0][2], current_pose[1]])
        future_pose = torch.Tensor([future_pose[0][0], future_pose[0][1], future_pose[0][2], future_pose[1]])

        return image, current_pose, future_pose


def draw_arm_pose(top_down_image, side_view_image, arm_pose, color, arm_lengths=[[0.23682, 0.03], 0.28015], scale=500):
    pose_rad = [arm_pose[0]*pi/180, pi/2 - arm_pose[1]*pi/180, -1*(arm_pose[2]*pi/180 - pi/2)] # Converting units to radians and more correct angles

    top_down_vecs = [Vector(0, cos(pose_rad[1])*arm_lengths[0][0]*scale), Vector(0, cos(pose_rad[1] - pi/2 + pose_rad[2])*arm_lengths[1]*scale)]
    rotation_mat = Matrix.generate_rotation_matrix(arm_pose[0])
    
    top_down_vecs[1] = top_down_vecs[1] + top_down_vecs[0]
    top_down_vecs[1] = rotation_mat*top_down_vecs[1]
    top_down_vecs[1] = top_down_vecs[1] - top_down_vecs[0]

    top_down_vecs[0] = rotation_mat*top_down_vecs[0]
    cv2.line(top_down_image, (int(top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1)), (int(top_down_vecs[0].x + top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1 - top_down_vecs[0].y)), (0.7*color[0], 0.7*color[1], 0.7*color[2]), 3)
    cv2.line(top_down_image, (int(top_down_vecs[0].x) + int(top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1 - top_down_vecs[0].y)), (int(top_down_vecs[0].x + top_down_image.shape[1]/2 + top_down_vecs[1].x), int(top_down_image.shape[0] - 1 - (top_down_vecs[0].y + top_down_vecs[1].y))), (0.5*color[0], 0.5*color[1], 0.5*color[2]), 3)
    cv2.circle(top_down_image, (int(top_down_vecs[0].x + top_down_image.shape[1]/2 + top_down_vecs[1].x), int(top_down_image.shape[0] - 1 - (top_down_vecs[0].y + top_down_vecs[1].y))), 3, (0, float(arm_pose[3]), 1 - float(arm_pose[3])), 2)

    side_vecs = [Vector(top_down_vecs[0].y, sin(pose_rad[1])*arm_lengths[0][0]*scale), Vector(top_down_vecs[1].y, sin(pose_rad[1] - pi/2 + pose_rad[2])*arm_lengths[1]*scale)]
    cv2.line(side_view_image, (0, int(side_view_image.shape[0] - 1)), (int(side_vecs[0].x), int(side_view_image.shape[0] - side_vecs[0].y)), (0.7*color[0], 0.7*color[1], 0.7*color[2]), 3)
    cv2.line(side_view_image, (int(side_vecs[0].x), int(side_view_image.shape[0] - side_vecs[0].y)), (int(side_vecs[0].x + side_vecs[1].x), int(side_view_image.shape[0] - (side_vecs[0].y + side_vecs[1].y))), (0.5*color[0], 0.5*color[1], 0.5*color[2]), 3)
    cv2.circle(side_view_image, (int(side_vecs[0].x + side_vecs[1].x), int(side_view_image.shape[0] - (side_vecs[0].y + side_vecs[1].y))), 3, (0, float(arm_pose[3]), 1 - float(arm_pose[3])), 2)

    return top_down_image, side_view_image

data = ArmEpisodeDataset("data/test")
image, current, future = data.__getitem__(18)

# print(current, future)
# print(torch.argmax(image[:, :, 3]))
top_image = np.zeros((500, 500, 3))
side_image = np.zeros((500, 500, 3))
draw_arm_pose(top_image, side_image, current, (1, 0, 0))
draw_arm_pose(top_image, side_image, future, (0, 0, 1))
print(current)
cv2.imshow("top", top_image)
cv2.imshow("side", side_image)
cv2.waitKey(0)

# def draw_arm_position(current_pose, future_pose, arm_lengths=[[0.23682, 0.03], 0.28015], scale=30):
#     current_pose_rad = [current_pose[0]*pi/180, current_pose[1]*pi/180, current_pose[2]*pi/180]
#     future_pose_rad = [future_pose[0]*pi/180, future_pose[1]*pi/180, future_pose[2]*pi/180]

#     current_arm_a_vec = Vector(0, arm_lengths[])
    