import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import datetime
import os.path as osp

# Import from the VAR repository
from models.vqvae import VQVAE

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

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    """
    Training function to be run by each process.
    """
    # Setup the distributed environment
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Download checkpoint if on master process (rank 0)
    MODEL_DEPTH = 12  # You may need to adjust this variable based on your needs
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
    
    if rank == 0:
        if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
        if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Using device: {device}")
    
    # Wait for rank 0 to download checkpoint
    dist.barrier()
    
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
        
        # Create deterministic split with global random seed
        generator = torch.Generator().manual_seed(42)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    else:
        # If train/val directories exist, use them
        train_dataset = PapDataset(train_dir, transform=train_transform)
        val_dataset = PapDataset(val_dir, transform=val_transform)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42
    )
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True
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
    if rank == 0:
        print(f"Loading pretrained weights from {vae_ckpt}")
    model.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # MSE loss for reconstruction
    mse_loss = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Set epoch for distributed samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0
        train_rec_loss = 0
        train_vq_loss = 0
        
        # Use tqdm only on rank 0
        train_iter = train_loader
        if rank == 0:
            train_iter = tqdm(train_loader)
            
        for batch_idx, data in enumerate(train_iter):
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
            
            # Update metrics (local)
            train_loss += loss.item()
            train_rec_loss += rec_loss.item()
            train_vq_loss += vq_loss.item()
            
            # Update progress bar on rank 0
            if rank == 0 and isinstance(train_iter, tqdm):
                train_iter.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")
        
        # Calculate average training losses (across all processes)
        train_loss_tensor = torch.tensor([train_loss / len(train_loader)], device=device)
        train_rec_loss_tensor = torch.tensor([train_rec_loss / len(train_loader)], device=device)
        train_vq_loss_tensor = torch.tensor([train_vq_loss / len(train_loader)], device=device)
        
        # Reduce losses across all GPUs
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_rec_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_vq_loss_tensor, op=dist.ReduceOp.SUM)
        
        train_loss = train_loss_tensor.item() / world_size
        train_rec_loss = train_rec_loss_tensor.item() / world_size
        train_vq_loss = train_vq_loss_tensor.item() / world_size
        
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
                
                # Update metrics (local)
                val_loss += loss.item()
                val_rec_loss += rec_loss.item()
                val_vq_loss += vq_loss.item()
        
        # Calculate average validation losses (across all processes)
        val_loss_tensor = torch.tensor([val_loss / len(val_loader)], device=device)
        val_rec_loss_tensor = torch.tensor([val_rec_loss / len(val_loader)], device=device)
        val_vq_loss_tensor = torch.tensor([val_vq_loss / len(val_loader)], device=device)
        
        # Reduce losses across all GPUs
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_rec_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_vq_loss_tensor, op=dist.ReduceOp.SUM)
        
        val_loss = val_loss_tensor.item() / world_size
        val_rec_loss = val_rec_loss_tensor.item() / world_size
        val_vq_loss = val_vq_loss_tensor.item() / world_size
        
        # Print epoch results (only for rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} (Rec: {train_rec_loss:.4f}, VQ: {train_vq_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} (Rec: {val_rec_loss:.4f}, VQ: {val_vq_loss:.4f})")
        
            # Save model if it's the best so far
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.save_dir, f"vqvae_pap_best.pth")
                # Save only the module state dict (not the DDP wrapper)
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved best model with loss {best_loss:.4f} to {save_path}")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(args.save_dir, f"vqvae_pap_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), checkpoint_path)
    
    # Clean up distributed environment
    cleanup()
    
    if rank == 0:
        print("Training complete!")
        print(f"Best validation loss: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Finetune VQVAE on PAP data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PAP dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers per GPU')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()
    
    # Determine world size (number of GPUs)
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    # Print multi-GPU training info
    print(f"Training with {args.world_size} GPUs")
    
    # Use multiprocessing to launch multiple processes
    mp.spawn(
        train_model,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()