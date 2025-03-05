import os
import datetime

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2
from math import cos, sin, pi
from matrix import *

from torch.utils.tensorboard import SummaryWriter

from arm_dataset import *
from arm_models import *

d_model = 16
img_size = (424, 240)
patch_size = (8, 8)
n_channels = 4
n_heads = 4
n_sa_blocks = 8
start_epoch = 44
n_epochs = 100
learning_rate = 0.001 # 0.005 -- works
val_every_n_epoch = 3
log_every_n_batches = 10
dataset_delta_frame = 10

train_batch_size = 20
test_batch_size = 1

PATH_TO_TRAIN_DATA = "data/train"
PATH_TO_TEST_DATA = "data/test"
PATH_TO_LOGS = "logs/run"
PATH_TO_MODEL = "logs/run_20250301_231222/armnet_43.pt"


def draw_arm_pose(top_down_image, side_view_image, arm_pose, color, arm_lengths=[[0.23682, 0.03], 0.28015], width=3, scale=500):
    pose_rad = [arm_pose[0]*pi/180, pi/2 - arm_pose[1]*pi/180, -1*(arm_pose[2]*pi/180 - pi/2)] # Converting units to radians and correct angles

    top_down_vecs = [Vector(0, cos(pose_rad[1])*arm_lengths[0][0]*scale), Vector(0, cos(pose_rad[1] - pi/2 + pose_rad[2])*arm_lengths[1]*scale)]
    rotation_mat = Matrix.generate_rotation_matrix(arm_pose[0])
    
    top_down_vecs[1] = top_down_vecs[1] + top_down_vecs[0]
    top_down_vecs[1] = rotation_mat*top_down_vecs[1]
    top_down_vecs[0] = rotation_mat*top_down_vecs[0]
    top_down_vecs[1] = top_down_vecs[1] - top_down_vecs[0]

    cv2.line(top_down_image, (int(top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1)), (int(top_down_vecs[0].x + top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1 - top_down_vecs[0].y)), (0.7*color[0], 0.7*color[1], 0.7*color[2]), width)
    cv2.line(top_down_image, (int(top_down_vecs[0].x) + int(top_down_image.shape[1]/2), int(top_down_image.shape[0] - 1 - top_down_vecs[0].y)), (int(top_down_vecs[0].x + top_down_image.shape[1]/2 + top_down_vecs[1].x), int(top_down_image.shape[0] - 1 - (top_down_vecs[0].y + top_down_vecs[1].y))), (0.5*color[0], 0.5*color[1], 0.5*color[2]), width)
    cv2.circle(top_down_image, (int(top_down_vecs[0].x + top_down_image.shape[1]/2 + top_down_vecs[1].x), int(top_down_image.shape[0] - 1 - (top_down_vecs[0].y + top_down_vecs[1].y))), 3, (0, float(arm_pose[3]), 1 - float(arm_pose[3])), 2)

    side_vecs = [Vector(top_down_vecs[0].y, sin(pose_rad[1])*arm_lengths[0][0]*scale), Vector(top_down_vecs[1].y, sin(pose_rad[1] - pi/2 + pose_rad[2])*arm_lengths[1]*scale)]
    cv2.line(side_view_image, (0, int(side_view_image.shape[0] - 100)), (int(side_vecs[0].x), int(side_view_image.shape[0] - side_vecs[0].y - 100)), (0.7*color[0], 0.7*color[1], 0.7*color[2]), width)
    cv2.line(side_view_image, (int(side_vecs[0].x), int(side_view_image.shape[0] - side_vecs[0].y - 100)), (int(side_vecs[0].x + side_vecs[1].x), int(side_view_image.shape[0] - (side_vecs[0].y + side_vecs[1].y) - 100)), (0.5*color[0], 0.5*color[1], 0.5*color[2]), width)
    cv2.circle(side_view_image, (int(side_vecs[0].x + side_vecs[1].x), int(side_view_image.shape[0] - (side_vecs[0].y + side_vecs[1].y) - 100)), 3, (0, float(arm_pose[3]), 1 - float(arm_pose[3])), 2)

    return top_down_image, side_view_image


def load_model(path):
    model = ArmTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_sa_blocks)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


# Main function
def main():
    # Create new log directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_path = f"{PATH_TO_LOGS}_{timestamp}"
    writer = SummaryWriter(log_dir=log_path)

    # Create datasets
    train_dataset = ArmEpisodeDataset(PATH_TO_TRAIN_DATA, delta_frame=dataset_delta_frame, image_transform=dataTransformations)
    test_dataset = ArmEpisodeDataset(PATH_TO_TEST_DATA, delta_frame=dataset_delta_frame, image_transform=dataTransformations)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # Create arm model
    arm_transformer_model = ArmTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_sa_blocks).to(device)
    if PATH_TO_MODEL != None :
        arm_transformer_model = load_model(PATH_TO_MODEL).to(device)

    optimizer = AdamW(arm_transformer_model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction='mean')

    # Train
    print("-- Starting training --")
    
    total_samples = 0
    iter_to_log = total_samples
    for epoch in range(start_epoch, n_epochs):
        arm_transformer_model.train()
        training_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            images, current_poses, future_poses = data
            images, current_poses, future_poses = images.to(device), current_poses.to(device), future_poses.to(device)

            optimizer.zero_grad()

            outputs = arm_transformer_model(images, current_poses)
            loss = criterion(outputs, future_poses)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if i % log_every_n_batches == 0:
                iter_to_log = total_samples

                # Log losses
                loss_to_log = training_loss/log_every_n_batches
                training_loss = 0.0
                writer.add_scalar("Loss", loss_to_log, total_samples)
                print(f"Training Loss: {loss_to_log}")

                # Log data
                image = images[0].detach().cpu().numpy()
                current_pose = current_poses[0].cpu()
                future_pose = future_poses[0].cpu()
                predicted_pose = outputs[0].cpu()

                current_pose[:3] = current_pose[:3]*180.0
                future_pose[:3] = future_pose[:3]*180.0
                predicted_pose[:3] = predicted_pose[:3]*180.0

                side_view = np.zeros((500, 500, 3))
                top_view = np.zeros((500, 500, 3))
                draw_arm_pose(top_view, side_view, current_pose, (1, 1, 1), width=12)
                draw_arm_pose(top_view, side_view, future_pose, (0, 1, 0), width=7)
                draw_arm_pose(top_view, side_view, predicted_pose, (1, 0, 0), width=3)

                writer.add_image("side view", transforms.ToTensor()(side_view), iter_to_log)
                writer.add_image("top view", transforms.ToTensor()(top_view), iter_to_log)
                writer.add_image("color", image[:3, :, :], iter_to_log)
                writer.add_image("depth", image[3, None, :, :], iter_to_log)
                writer.flush()

            total_samples += train_batch_size

        # Save the model checkpoint for this epoch
        print(f'Epoch {epoch}/{n_epochs}')
        torch.save(arm_transformer_model.state_dict(), os.path.join(log_path, f"armnet_{epoch}.pt")) 

        # Validate 
        if epoch % val_every_n_epoch == 0:
            arm_transformer_model.eval()  # Set the model to evaluation mode
            val_total_loss = 0.0

            with torch.no_grad():   # Disable gradient calculation
                
                for data in test_dataloader:
                    images, current_poses, future_poses = data
                    images, current_poses, future_poses = images.to(device), current_poses.to(device), future_poses.to(device)

                    outputs = arm_transformer_model(images, current_poses)
                    loss = criterion(outputs, future_poses)
                    val_total_loss += loss.item()

                validation_loss = val_total_loss / len(test_dataloader)
                print(f"Validation Loss: {validation_loss}")
                writer.add_scalar("Validation Loss", validation_loss, iter_to_log)


# __name__
if __name__=="__main__":
    main()


