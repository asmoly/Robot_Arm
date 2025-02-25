import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import cv2

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels

        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.pred_pose_token = nn.Parameter(torch.randn(1, 1, d_model)) # pose prediction token
        self.current_pose_token = torch.Tensor((1, 1, d_model)) # pose prediction token
        
        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.pred_pose_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe

        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1)

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out
  

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))

        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))

        return out


class VisionTransformer(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model # Dimensionality of model
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_layers)])

        # Input pose mapper
        self.pose_to_token = nn.Sequential(
            nn.Linear(4, self.d_model), # 4 is the number of joint angles on the arm
        )

        # Pose predictor MLP
        self.pose_predictor = nn.Sequential(
            nn.Linear(self.d_model, 4), # 4 is the number of joint angles on the arm
        )

    def forward(self, images, current_poses):
        x = self.patch_embedding(images)

        # create tokens from current_poses and concatenate them to x
        # create random init tokens for pred_poses and concatenate them to x

        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x)
        
        x = self.pose_predictor(x[:,0])

        return x
    
class ArmEpisodeDataset(Dataset):
    def __init__(self, path_to_data, image_transform=None):
        self.path_to_data = path_to_data
        self.image_transform = image_transform

        print(f"Loading data from {self.path_to_data}")
        self.episodes = []
        for episode in os.listdir(self.path_to_data):
            episode_length = len([f for f in os.listdir(os.path.join(self.path_to_data, episode, "color_images")) if os.path.isfile(os.path.join(os.path.join(self.path_to_data, episode, "color_images"), f))])
            self.episodes.append([episode, episode_length])

        self.data_length = 0
        for episode in self.episodes:
            self.data_length += (episode[1] - 1) # We dont take into account the last frame as there is nothing after it

        print(f"Finished loading data from {self.path_to_data}")

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        frame_count = 0
        for episode in self.episodes:
            if idx <= (frame_count + episode[1] - 1):
                print(os.path.join(self.path_to_data, episode[0], "color_images", f"{str(idx - frame_count)}.png"))
                color_image = cv2.imread(os.path.join(self.path_to_data, episode[0], "color_images", f"{str(idx - frame_count)}.png"))
                depth_image = cv2.imread(os.path.join(self.path_to_data, episode[0], "depth_images", f"{str(idx - frame_count)}.png"))
                current_arm_position = pickle.load(open(os.path.join(self.path_to_data, episode[0], "arm_positions", str(idx - frame_count)), "rb"))
                future_arm_position = pickle.load(open(os.path.join(self.path_to_data, episode[0], "arm_positions", str(idx - frame_count + 1)), "rb")) # +1 because it is the next position
                
                color_image = torch.Tensor(color_image)
                if self.image_transform != None:
                    color_image = self.image_transform(torch.Tensor(color_image))

                depth_image = torch.Tensor(depth_image)

                image = torch.zeros((480, 848, 4))
                image[:, :, :3] = color_image
                image[:, :, 3] = depth_image
                
                current_arm_position = torch.Tensor([current_arm_position[0][0], current_arm_position[0][1], current_arm_position[0][2], current_arm_position[1]])
                future_arm_position = torch.Tensor([future_arm_position[0][0], future_arm_position[0][1], future_arm_position[0][2], future_arm_position[1]])

                return image, current_arm_position, future_arm_position
            
            frame_count += (episode[1] - 1)
  
d_model = 9
img_size = (848, 480)
patch_size = (16, 16)
n_channels = 4
n_heads = 3
n_layers = 3
epochs = 5
alpha = 0.005

train_batch_size = 10
test_batch_size = 1

PATH_TO_TRAIN_DATA = "data/train"
PATH_TO_TEST_DATA = "data/test"

train_dataset = ArmEpisodeDataset(PATH_TO_TRAIN_DATA)
test_dataset = ArmEpisodeDataset(PATH_TO_TEST_DATA)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

transformer = VisionTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

optimizer = Adam(transformer.parameters(), lr=alpha)
criterion = nn.L1Loss()

print("-- Starting training --")
for epoch in range(epochs):
    training_loss = 0.0

    for i, data in enumerate(train_dataloader, 0):
        image, current_arm_pos, future_arm_pos = data
        image, current_arm_pos, future_arm_pos = image.to(device), current_arm_pos.to(device), future_arm_pos.to(device)

        optimizer.zero_grad()

        outputs = transformer(image)
        loss = criterion(outputs, future_arm_pos)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = transformer(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'\nModel Accuracy: {100 * correct // total} %')