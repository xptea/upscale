import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

class SRDatasetMultiScale(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None, target_size=(256, 256)):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.target_size = target_size
        self.pairs = []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
        high_files = sorted([f for f in os.listdir(high_res_dir) if os.path.splitext(f)[1] in image_extensions])
        low_files = sorted([f for f in os.listdir(low_res_dir) if os.path.splitext(f)[1] in image_extensions])
        
        for high_file in high_files:
            base_name = os.path.splitext(high_file)[0]
            
            for scale in ['_2', '_4', '_6']:
                low_file = base_name + scale + os.path.splitext(high_file)[1]
                if low_file in low_files:
                    self.pairs.append((low_file, high_file))
        
        if len(self.pairs) == 0:
            raise ValueError(f"No matching image pairs found in {low_res_dir} and {high_res_dir}")
        
        print(f"Found {len(self.pairs)} image pairs with scales (_2, _4, _6)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_file, high_file = self.pairs[idx]
        low_path = os.path.join(self.low_res_dir, low_file)
        high_path = os.path.join(self.high_res_dir, high_file)
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        low_img = TF.resize(low_img, self.target_size, interpolation=TF.InterpolationMode.BICUBIC)
        high_img = TF.resize(high_img, self.target_size, interpolation=TF.InterpolationMode.BICUBIC)
        
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        
        return low_img, high_img
