import os
import numpy as np
import pickle
import cv2
from math import cos, sin, pi
from matrix import *
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

REALSENSE_INF = 65535 # in units of 0.01cm

dataTransformations = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.0)),
])

class ArmEpisodeDataset(Dataset):
    def __init__(self, path_to_data, delta_frame=1, image_transform=None):
        self.path_to_data = path_to_data
        self.image_transform = image_transform
        self.delta_frame = delta_frame

        self.frames = []
        for episode_dir in os.listdir(self.path_to_data):
            full_path = os.path.join(self.path_to_data, episode_dir)
            
            # TODO: why do we need walk?
            episode_frame_count = 0
            for root, dirs, files in os.walk(os.path.join(full_path, "color_images")):
                episode_frame_count += len(files)

            # TODO: assign listdir to a temp var list and then use
            for frame in os.listdir(os.path.join(full_path, "color_images")):
                if int(frame.replace(".png", "")) < (episode_frame_count - self.delta_frame):
                    self.frames.append([full_path, int(frame.replace(".png", ""))])

        self.data_length = len(self.frames)

    @staticmethod
    def normalize_data(image, current_pose, future_pose, max_depth=REALSENSE_INF):
        current_pose[:3] = current_pose[:3]/180.0 # range of motion of motors is 180 degrees
        future_pose[:3] = future_pose[:3]/180.0   
        image[:, :, :3] = image[:, :, :3]/255.0  # normalizes rgb values
        image[:, :, 3] = image[:, :, 3]/max_depth   # normalizes depth

        return image, current_pose, future_pose

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        path_to_color_image = os.path.join(frame_path[0], "color_images", f"{frame_path[1]}.png")
        path_to_depth_image = os.path.join(frame_path[0], "depth_images", f"{frame_path[1]}.png")
        path_to_current_pose = os.path.join(frame_path[0], "arm_positions", f"{frame_path[1]}")
        path_to_future_pose = os.path.join(frame_path[0], "arm_positions", f"{frame_path[1] + self.delta_frame}")

        color_image = cv2.imread(path_to_color_image).astype(float)
        depth_image = cv2.imread(path_to_depth_image, cv2.IMREAD_UNCHANGED).astype(float)
        current_pose = pickle.load(open(path_to_current_pose, "rb"))
        future_pose = pickle.load(open(path_to_future_pose, "rb"))

        color_image = cv2.resize(color_image, (int(color_image.shape[1]/2), int(color_image.shape[0]/2)))
        depth_image = cv2.resize(depth_image, (int(depth_image.shape[1]/2), int(depth_image.shape[0]/2)))
                
        color_image = torch.Tensor(color_image)
        depth_image = torch.Tensor(depth_image)

        image = torch.zeros((color_image.shape[0], color_image.shape[1], 4))
        image[:, :, :3] = color_image
        image[:, :, 3] = depth_image
        
        current_pose = torch.Tensor([current_pose[0][0], current_pose[0][1], current_pose[0][2], current_pose[1]])
        future_pose = torch.Tensor([future_pose[0][0], future_pose[0][1], future_pose[0][2], future_pose[1]])

        image, current_pose, future_pose = ArmEpisodeDataset.normalize_data(image, current_pose, future_pose)

        image = torch.permute(image, (2, 0, 1))

        if self.image_transform != None:
            image[:3, :, :] = self.image_transform(image[:3, :, :])

        return image, current_pose, future_pose
    