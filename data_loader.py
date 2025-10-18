import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

class SRDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
        low_files = sorted([f for f in os.listdir(low_res_dir) if os.path.splitext(f)[1] in image_extensions])
        high_files = sorted([f for f in os.listdir(high_res_dir) if os.path.splitext(f)[1] in image_extensions])
        
        low_set = set(low_files)
        high_set = set(high_files)
        matching_files = sorted(low_set & high_set)
        
        if len(matching_files) == 0:
            raise ValueError(f"No matching image files found in {low_res_dir} and {high_res_dir}")
        
        self.low_files = matching_files
        self.high_files = matching_files
        self.transform = transform
        print(f"Found {len(self.low_files)} matching image pairs")

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_res_dir, self.low_files[idx])
        high_path = os.path.join(self.high_res_dir, self.high_files[idx])
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        low_img = TF.resize(low_img, high_img.size[::-1], interpolation=TF.InterpolationMode.BICUBIC)
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        return low_img, high_img