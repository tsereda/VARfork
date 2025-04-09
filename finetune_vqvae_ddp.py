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

# Import necessary libraries for distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import from the VAR repository
from models.vqvae import VQVAE

# --- No longer need the SimpleDist mock ---

# Helper function for distributed setup
def setup_distributed(backend="nccl", port=None):
    """Initializes the distributed environment."""
    if dist.is_initialized():
        return
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # If MASTER_ADDR and MASTER_PORT are not set, try to infer them if possible
    # (This is often handled by launchers like torchrun or SLURM)
    if world_size > 1:
        print(f"Initializing distributed process group (backend: {backend})")
        if os.environ.get("MASTER_ADDR") is None:
            print("Warning: MASTER_ADDR not set. Using default localhost.")
            os.environ["MASTER_ADDR"] = "localhost"
        if os.environ.get("MASTER_PORT") is None:
            print("Warning: MASTER_PORT not set. Using default 29500.")
            os.environ["MASTER_PORT"] = port if port else "29500" # Default port
            
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        print(f"Distributed setup complete: Rank {rank}/{world_size}, Local Rank {local_rank}")
    else:
        print("Running in non-distributed mode (world_size=1)")
        
    # Synchronize before starting
    if world_size > 1:
        dist.barrier()

def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed environment cleaned up.")

def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Gets the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """Gets the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

# Define PAP dataset loader (remains the same)
class PapDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        
        # Find all image files (only rank 0 needs to scan initially if files are shared)
        # However, each process needs the list, so let all do it unless it's very slow.
        # If very slow, have rank 0 scan and broadcast the list.
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    self.img_paths.append(os.path.join(root, file))
        
        if is_main_process():
             print(f"Found {len(self.img_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Ensure idx is within bounds for the current process's view via the sampler
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path} on rank {get_rank()}: {e}")
            # Return a placeholder tensor or skip; here returning None and handling in collate_fn might be better
            # For simplicity, returning a dummy tensor of the expected size
            return torch.zeros((3, 256, 256)) # Adjust size if needed


# Normalization function (remains the same)
def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1]
    return x.add(x).add_(-1)

# Function to reduce loss across all GPUs
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def main():
    parser = argparse.ArgumentParser(description='Finetune VQVAE on PAP data with DDP')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PAP dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size *per GPU*')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ddp', help='Directory to save models')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers per GPU')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank. Passed by launch script.') # Added for DDP
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend (nccl recommended for Nvidia GPUs)')
    parser.add_argument('--dist_port', type=str, default='29500', help='Port for master node communication')

    args = parser.parse_args()
    
    # --- Distributed Setup ---
    setup_distributed(backend=args.dist_backend, port=args.dist_port)
    world_size = get_world_size()
    rank = get_rank()
    device = torch.device(f'cuda:{args.local_rank}')
    
    if is_main_process():
        print(f"Starting DDP training with {world_size} GPUs.")
        print(f"Effective batch size: {args.batch_size * world_size}")
        # Create save directory only on main process
        os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Download Checkpoint (only on main process) ---
    MODEL_DEPTH = 12 # Adjust if needed
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

    if is_main_process():
        print("Downloading checkpoints if necessary (main process only)...")
        if not osp.exists(vae_ckpt):
            print(f"Downloading {vae_ckpt}...")
            os.system(f'wget {hf_home}/{vae_ckpt}')
        # if not osp.exists(var_ckpt): # VAR checkpoint not used in this script
        #     print(f"Downloading {var_ckpt}...")
        #     os.system(f'wget {hf_home}/{var_ckpt}')
        print("Checkpoint download check complete.")
            
    # --- Ensure all processes wait for rank 0 to finish downloads ---
    if world_size > 1:
        dist.barrier()
        if is_main_process():
             print("All processes synchronized after checkpoint check.")
        
    # --- Transforms (remain the same) ---
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
    
    # --- Datasets ---
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        if is_main_process():
            print("Train/Val directories not found. Creating 80/20 split from root data path.")
        # If no train/val split, use the main directory and create a split
        full_dataset = PapDataset(args.data_path, transform=None) # Load metadata first
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Use random split - ensure generator seed is same across processes initially
        train_indices, val_indices = torch.utils.data.random_split(
            range(total_size), [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        ).indices # Get indices
        
        # Create subset datasets based on indices
        train_dataset_full = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset_full = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Assign transforms AFTER splitting
        # Need a way to set transform on the underlying dataset within Subset
        # Option 1: Custom Subset class or modify PapDataset to take indices
        # Option 2: Apply transform within the loop (less ideal)
        # Option 3 (Chosen): Re-instantiate PapDataset with transform for subsets (cleaner)
        train_dataset = PapDataset(args.data_path, transform=train_transform)
        val_dataset = PapDataset(args.data_path, transform=val_transform)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    else:
        if is_main_process():
             print("Using existing train/val directories.")
        # If train/val directories exist, use them
        train_dataset = PapDataset(train_dir, transform=train_transform)
        val_dataset = PapDataset(val_dir, transform=val_transform)
    
    # --- Distributed Samplers ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # --- Data Loaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, # Use sampler instead of shuffle=True
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=True # Often recommended for DDP
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, # Use sampler
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False # Keep all validation samples
    )
    
    # --- Create Model ---
    model = VQVAE(
        vocab_size=4096, ch=160, z_channels=32, quant_resi=0.5, share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), test_mode=False
    ).to(device)
    
    # --- Load Pretrained Weights (Load on CPU first then move to device to save GPU memory) ---
    if is_main_process():
        print(f"Loading pretrained weights from {vae_ckpt}")
    # Load to CPU first on all processes
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank} if torch.cuda.is_available() else 'cpu' # Map to current device
    checkpoint = torch.load(vae_ckpt, map_location='cpu') 
    # Adapt loading if state_dict structure differs (e.g. if it was saved with DDP 'module.' prefix)
    # For checkpoints saved *without* DDP, direct loading should be fine.
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed ({e}). Trying without strict.")
        # Handle potential 'module.' prefix if loading a DDP checkpoint into non-DDP or vice-versa
        state_dict = checkpoint
        if not hasattr(model, 'module') and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
        elif hasattr(model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
             state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)

    model.to(device) # Move model to assigned GPU

    # --- Wrap Model with DDP ---
    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False) # Set find_unused_parameters=True if you encounter issues
        if is_main_process():
            print("Model wrapped with DistributedDataParallel.")
    
    # --- Optimizer ---
    # Note: Adam uses buffers that might need syncing if using certain schedulers or techniques.
    # Basic Adam setup should be fine. Scale LR potentially? (e.g., linear scaling rule: args.lr * world_size)
    # Keeping original LR for simplicity here.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Loss Function ---
    mse_loss = nn.MSELoss().to(device) # Ensure loss is on the correct device
    
    # --- Training Loop ---
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        train_rec_loss_accum = 0.0
        train_vq_loss_accum = 0.0
        num_train_batches = 0
        
        # Wrap train_loader with tqdm only on the main process
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=not is_main_process())
        
        for batch_idx, data in enumerate(pbar):
            # Handle potential tuple data (image, label) -> image
            if isinstance(data, (tuple, list)):
                data = data[0]
                
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            # Note: DDP automatically handles gradient synchronization during backward()
            recon, usages, vq_loss = model(data)
            
            # Calculate reconstruction loss
            rec_loss = mse_loss(recon, data)
            
            # Total loss (ensure vq_loss is a scalar tensor on the correct device)
            if not isinstance(vq_loss, torch.Tensor): # Handle potential non-tensor return
                vq_loss = torch.tensor(vq_loss, device=device, dtype=rec_loss.dtype)
            elif vq_loss.dim() > 0: # Ensure scalar
                vq_loss = vq_loss.mean() 

            loss = rec_loss + vq_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # --- Accumulate losses for reporting ---
            # Reduce losses across GPUs for accurate average reporting
            if world_size > 1:
                reduced_loss = reduce_tensor(loss.detach(), world_size)
                reduced_rec_loss = reduce_tensor(rec_loss.detach(), world_size)
                reduced_vq_loss = reduce_tensor(vq_loss.detach(), world_size)
            else:
                reduced_loss = loss.detach()
                reduced_rec_loss = rec_loss.detach()
                reduced_vq_loss = vq_loss.detach()

            train_loss_accum += reduced_loss.item()
            train_rec_loss_accum += reduced_rec_loss.item()
            train_vq_loss_accum += reduced_vq_loss.item()
            num_train_batches += 1
            
            # Update progress bar on main process
            if is_main_process():
                 pbar.set_postfix({
                    "Loss": f"{reduced_loss.item():.4f}", 
                    "Rec": f"{reduced_rec_loss.item():.4f}", 
                    "VQ": f"{reduced_vq_loss.item():.4f}"
                 })
                 
        # Calculate average training losses for the epoch
        avg_train_loss = train_loss_accum / num_train_batches
        avg_train_rec_loss = train_rec_loss_accum / num_train_batches
        avg_train_vq_loss = train_vq_loss_accum / num_train_batches
        
        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        val_rec_loss_accum = 0.0
        val_vq_loss_accum = 0.0
        num_val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=not is_main_process())
        
        with torch.no_grad():
            for data in val_pbar:
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.to(device)
                
                # Forward pass
                recon, usages, vq_loss = model(data)
                
                rec_loss = mse_loss(recon, data)
                
                if not isinstance(vq_loss, torch.Tensor):
                    vq_loss = torch.tensor(vq_loss, device=device, dtype=rec_loss.dtype)
                elif vq_loss.dim() > 0:
                     vq_loss = vq_loss.mean()

                loss = rec_loss + vq_loss
                
                # Reduce losses across GPUs
                if world_size > 1:
                    reduced_val_loss = reduce_tensor(loss.detach(), world_size)
                    reduced_val_rec_loss = reduce_tensor(rec_loss.detach(), world_size)
                    reduced_val_vq_loss = reduce_tensor(vq_loss.detach(), world_size)
                else:
                    reduced_val_loss = loss.detach()
                    reduced_val_rec_loss = rec_loss.detach()
                    reduced_val_vq_loss = vq_loss.detach()
                
                val_loss_accum += reduced_val_loss.item()
                val_rec_loss_accum += reduced_val_rec_loss.item()
                val_vq_loss_accum += reduced_val_vq_loss.item()
                num_val_batches += 1

                if is_main_process():
                    val_pbar.set_postfix({
                       "Val Loss": f"{reduced_val_loss.item():.4f}" 
                    })

        # Calculate average validation losses
        avg_val_loss = val_loss_accum / num_val_batches
        avg_val_rec_loss = val_rec_loss_accum / num_val_batches
        avg_val_vq_loss = val_vq_loss_accum / num_val_batches
        
        # Print epoch results (only on main process)
        if is_main_process():
            print(f"\n--- Epoch {epoch+1}/{args.epochs} Summary ---")
            print(f"Train Loss: {avg_train_loss:.4f} (Rec: {avg_train_rec_loss:.4f}, VQ: {avg_train_vq_loss:.4f})")
            print(f"Val Loss:   {avg_val_loss:.4f} (Rec: {avg_val_rec_loss:.4f}, VQ: {avg_val_vq_loss:.4f})")
            
            # --- Save model (only on main process) ---
            # Save best model based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_path = os.path.join(args.save_dir, "vqvae_pap_best.pth")
                # Save the underlying model's state_dict (unwrap DDP)
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), save_path)
                print(f"Saved best model with val_loss {best_loss:.4f} to {save_path}")
            
            # Save checkpoint after each epoch (optional)
            # checkpoint_path = os.path.join(args.save_dir, f"vqvae_pap_epoch_{epoch+1}.pth")
            # model_to_save = model.module if isinstance(model, DDP) else model
            # torch.save(model_to_save.state_dict(), checkpoint_path)
            # print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
            print("-" * (len("--- Epoch / Summary ---") + 2 * len(str(args.epochs)))) # Separator

    # --- Final Messages ---
    if is_main_process():
        print("\nTraining complete!")
        print(f"Best validation loss achieved: {best_loss:.4f}")

    # --- Cleanup Distributed Environment ---
    cleanup_distributed()

if __name__ == "__main__":
    # Important: This script should be launched using torchrun or torch.distributed.launch
    # Example: torchrun --standalone --nnodes=1 --nproc_per_node=2 finetune_vqvae_ddp.py --data_path /path/to/pap --batch_size 32 --epochs 50 
    main()