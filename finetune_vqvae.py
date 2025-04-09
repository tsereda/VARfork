import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import datetime
import os.path as osp

# Import from the VAR repository
from models.vqvae import VQVAE

# Initialize a simple dist module replacement
class SimpleDist:
    @staticmethod
    def get_world_size():
        return 1
    
    @staticmethod
    def initialized():
        return False

# Apply the patch
import sys
import models.quant
sys.modules['dist'] = SimpleDist()
models.quant.dist = SimpleDist()
models.quant.tdist = SimpleDist()

# Define PAP dataset loader
class PapDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        
        # Find all image files
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    self.img_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.img_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Normalization function
def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1]
    return x.add(x).add_(-1)

def main():
    parser = argparse.ArgumentParser(description='Finetune VQVAE on PAP data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PAP dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    
    # Download checkpoint
    MODEL_DEPTH = 12  # You may need to adjust this variable based on your needs
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    
    # Create datasets
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    
    # Check if the dataset has a specific structure
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # If no train/val split, use the main directory and create a split
        dataset = PapDataset(args.data_path, transform=None)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Use random split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    else:
        # If train/val directories exist, use them
        train_dataset = PapDataset(train_dir, transform=train_transform)
        val_dataset = PapDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    # Create model
    model = VQVAE(
        vocab_size=4096,  # Default
        z_channels=32,    # Default
        ch=160,           # Default
        quant_resi=0.5,   # Default
        share_quant_resi=4,  # Default 
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # Default
        test_mode=False   # Set to False for training
    ).to(device)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {vae_ckpt}")
    model.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)
    
    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # MSE loss for reconstruction
    mse_loss = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_rec_loss = 0
        train_vq_loss = 0
        
        pbar = tqdm(train_loader)
        for batch_idx, data in enumerate(pbar):
            # If data is a tuple (image, label), take only the image
            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]
                
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, usages, vq_loss = model(data)
            
            # Calculate reconstruction loss
            rec_loss = mse_loss(recon, data)
            
            # Total loss is reconstruction loss + VQ loss
            loss = rec_loss + vq_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_rec_loss += rec_loss.item()
            train_vq_loss += vq_loss.item()
            
            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")
        
        # Calculate average training losses
        train_loss /= len(train_loader)
        train_rec_loss /= len(train_loader)
        train_vq_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_rec_loss = 0
        val_vq_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                # If data is a tuple (image, label), take only the image
                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]
                    
                data = data.to(device)
                
                # Forward pass
                recon, usages, vq_loss = model(data)
                
                # Calculate reconstruction loss
                rec_loss = mse_loss(recon, data)
                
                # Total loss
                loss = rec_loss + vq_loss
                
                # Update metrics
                val_loss += loss.item()
                val_rec_loss += rec_loss.item()
                val_vq_loss += vq_loss.item()
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        val_rec_loss /= len(val_loader)
        val_vq_loss /= len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} (Rec: {train_rec_loss:.4f}, VQ: {train_vq_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Rec: {val_rec_loss:.4f}, VQ: {val_vq_loss:.4f})")
        
        # Save model if it's the best so far
        if val_loss < best_loss:
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.save_dir, f"vqvae_pap_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with loss {best_loss:.4f} to {save_path}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.save_dir, f"vqvae_pap_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
    
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()