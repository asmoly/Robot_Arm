from torch.utils.data import DataLoader, Dataset
import torch
import os
import cv2
import numpy as np
import pickle

class ArmEpisodeData(Dataset):
    def __init__(self, path_to_data, image_transform=None):
        self.path_to_data = path_to_data
        self.image_transform = image_transform

        self.episodes = []
        for episode in os.listdir(self.path_to_data):
            episode_length = len([f for f in os.listdir(os.path.join(self.path_to_data, episode, "color_images")) if os.path.isfile(os.path.join(os.path.join(self.path_to_data, episode, "color_images"), f))])
            self.episodes.append([episode, episode_length])

        self.data_length = 0
        for episode in self.episodes:
            self.data_length += (episode[1] - 1) # We dont take into account the last frame as there is nothing after it

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        frame_count = 0
        for episode in self.episodes:
            if idx <= (frame_count + episode[1] - 1):
                print(os.path.join(self.path_to_data, episode[0], "color_images", f"{str(idx - frame_count)}.png"))
                color_image = cv2.imread(os.path.join(self.path_to_data, episode[0], "color_images", f"{str(idx - frame_count)}.png"))
                depth_image = cv2.imread(os.path.join(self.path_to_data, episode[0], "depth_images", f"{str(idx - frame_count)}.png"))
                arm_position = pickle.load(open(os.path.join(self.path_to_data, episode[0], "arm_positions", str(idx - frame_count)), "rb"))
                
                color_image = torch.Tensor(color_image)
                if self.image_transform != None:
                    color_image = self.image_transform(torch.Tensor(color_image))

                depth_image = torch.Tensor(depth_image)
                arm_position = torch.Tensor([arm_position[0][0], arm_position[0][1], arm_position[0][2], 1])

                return color_image, depth_image, arm_position
            
            frame_count += (episode[1] - 1)


data = ArmEpisodeData("data/test")
print(data.__len__())
color, depth, arm = data.__getitem__(1000)
print(arm)
