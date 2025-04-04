import os
import shutil
from pathlib import Path
import glob

# Source and destination paths
src_root = Path("/opt/repo/VARfork/pap_data/cyclegan_dataset_256_split")
dst_root = Path("/opt/repo/VARfork/pap_data_formatted")

# Create destination directories
os.makedirs(dst_root / "train" / "class0", exist_ok=True)
os.makedirs(dst_root / "train" / "class1", exist_ok=True)
os.makedirs(dst_root / "val" / "class0", exist_ok=True)
os.makedirs(dst_root / "val" / "class1", exist_ok=True)

# Let's first check what file extensions exist in the source directories
for dir_name in ["trainA", "trainB", "valA", "valB"]:
    path = src_root / dir_name
    if not path.exists():
        print(f"Warning: {path} does not exist!")
        continue
        
    # Print the first few files to check extensions
    files = list(path.glob("*"))
    if not files:
        print(f"Warning: No files found in {path}")
    else:
        print(f"Found {len(files)} files in {path}")
        print(f"Sample files: {[f.name for f in files[:5]]}")
        
        # Now copy the files (any extension)
        if dir_name == "trainA":
            for file in files:
                shutil.copy(file, dst_root / "train" / "class0" / file.name)
            print(f"Copied {len(files)} files from {dir_name} to train/class0")
        elif dir_name == "trainB":
            for file in files:
                shutil.copy(file, dst_root / "train" / "class1" / file.name)
            print(f"Copied {len(files)} files from {dir_name} to train/class1")
        elif dir_name == "valA":
            for file in files:
                shutil.copy(file, dst_root / "val" / "class0" / file.name)
            print(f"Copied {len(files)} files from {dir_name} to val/class0")
        elif dir_name == "valB":
            for file in files:
                shutil.copy(file, dst_root / "val" / "class1" / file.name)
            print(f"Copied {len(files)} files from {dir_name} to val/class1")

print("Dataset reorganization attempt completed!")
