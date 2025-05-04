import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import glob

class ImageNoiseDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.*")))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        noise = torch.randn_like(img)
        return noise, img