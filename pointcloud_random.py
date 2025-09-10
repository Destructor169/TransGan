import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import numpy as np

# Transformer Block for both Generator and Discriminator
class TransformerBlock3D(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock3D, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))  # Use F.relu
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Generator for 3D Point Clouds
class PointCloudGenerator(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, input_dim=100, output_dim=3, num_points=1024):
        super(PointCloudGenerator, self).__init__()
        self.num_points = num_points
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_points, d_model))
        self.transformer_layers = nn.ModuleList([
            TransformerBlock3D(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, z):
        # z: latent code, shape (batch_size, input_dim)
        batch_size = z.size(0)
        x = self.input_projection(z).unsqueeze(1)  # (batch_size, 1, d_model)
        x = x.repeat(1, self.num_points, 1)  # (batch_size, num_points, d_model)
        x = x + self.positional_encoding  # Add positional encoding
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_projection(x)  # Project to 3D coordinates
        return x  # Output shape: (batch_size, num_points, output_dim)

# Discriminator for 3D Point Clouds
class PointCloudDiscriminator(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, input_dim=3):
        super(PointCloudDiscriminator, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock3D(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: point cloud, shape (batch_size, num_points, input_dim)
        x = self.input_projection(x)  # Project to d_model
        for layer in self.transformer_layers:
            x = layer(x)
        # Global feature pooling (mean over points)
        x = x.mean(dim=1)
        x = self.output_projection(x)  # Project to scalar (real/fake score)
        return x

# Hyperparameters
d_model = 64
nhead = 4
num_layers = 6
dim_feedforward = 128
latent_dim = 100
num_points = 1024
batch_size = 16
learning_rate = 0.0002
num_epochs = 25

# Create Generator and Discriminator models
generator = PointCloudGenerator(d_model, nhead, num_layers, dim_feedforward, input_dim=latent_dim, num_points=num_points)
discriminator = PointCloudDiscriminator(d_model, nhead, num_layers, dim_feedforward)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Visualize Point Cloud using Open3D
def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# Training Loop
for epoch in range(num_epochs):
    for _ in range(batch_size):
        # Generate random latent vectors
        z = torch.randn(batch_size, latent_dim)

        # Generate fake point clouds
        fake_point_clouds = generator(z)

        # Create real point clouds (sampled from a dataset, using random points for now)
        real_point_clouds = torch.rand(batch_size, num_points, 3)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_scores = discriminator(real_point_clouds)
        fake_scores = discriminator(fake_point_clouds.detach())
        d_loss_real = criterion(real_scores, real_labels)
        d_loss_fake = criterion(fake_scores, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_scores = discriminator(fake_point_clouds)
        g_loss = criterion(fake_scores, real_labels) 
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


with torch.no_grad():
    z = torch.randn(1, latent_dim)
    generated_point_cloud = generator(z).squeeze(0).cpu().numpy()
    visualize_point_cloud(generated_point_cloud)