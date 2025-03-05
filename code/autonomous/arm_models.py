import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
from math import cos, sin, pi
from matrix import *


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels

        # OLD: self.conv2d_1 = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.conv2d_1 = nn.Conv2d(self.n_channels, 8, kernel_size=5, stride=1, padding=2)
        self.conv2d_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv2d_3 = nn.Conv2d(16, self.d_model, kernel_size=self.patch_size[0], stride=self.patch_size[0]//2)

    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    def forward(self, x):
        # OLD: x = self.conv2d_1(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = self.conv2d_1(x) # 212, 120
        x = nn.GELU()(x)
        x = self.conv2d_2(x)
        x = nn.GELU()(x)
        x = self.conv2d_3(x)

        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
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

        # Normalize
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        # Compute new token values
        token_values = attention @ V

        return token_values


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


class ArmTransformer(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_sa_blocks):
        super().__init__()        

        # OLD: assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert img_size[0] % 2 == 0 and img_size[1] % 2 == 0 and patch_size[0] % 2 == 0 and patch_size[1] % 2 == 0
        assert img_size[0] > patch_size[0] and img_size[1] > patch_size[1]
        assert (img_size[0]/2)%(patch_size[0]/2) == 0 and (img_size[1]/2)%(patch_size[1]/2) == 0
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model # Dimensionality of model
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads

        # OLD: self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        # img_size[0] is width, img_size[1] is height
        self.n_patches = int(((self.img_size[0]/2)/(self.patch_size[0]/2) - 1) * ((self.img_size[1]/2)/(self.patch_size[1]/2) - 1)) # -1 because last patch cannot be moved beyond border
        self.max_seq_length = self.n_patches + 2 # add 2 extra tokens containing input and predicted arm poses
        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)

        self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_sa_blocks)])

        # Input pose mapper
        self.pose_to_token = nn.Sequential(
            nn.Linear(4, self.d_model), # 4 is the number of joint angles on the arm
        )

        # Pose predictor MLP
        self.pose_predictor = nn.Sequential(
            nn.Linear(self.d_model, 4), # 4 is the number of joint angles on the arm
        )

        self.pred_pose_token = nn.Parameter(torch.randn(1, 1, d_model)) # pose prediction token
        self.current_pose_token = torch.Tensor((1, 1, d_model)) # pose prediction token

    def forward(self, images, current_poses):
        x = self.patch_embedding(images)

        # Expand to have a predicted token for every frame in a batch
        pred_pose_tokens = self.pred_pose_token.expand(x.size()[0], -1, -1)
        
        current_pose_tokens = self.pose_to_token(current_poses)
        #torch.unsqueeze(current_pose_tokens, 1)
        current_pose_tokens = current_pose_tokens[:, None, :]

        # Adding special pose tokens to the beginning of each embedding
        x = torch.cat((current_pose_tokens,x), dim=1)
        x = torch.cat((pred_pose_tokens,x), dim=1)
        
        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x)
        
        x = self.pose_predictor(x[:,0]) # take 0-th token which must be a predicted token which we created at 0-th index above!

        return x
    