import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
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

# Now create the dataset
data_path = "data"  # Path to your image folder
full = ImageNoiseDataset(data_path)
print(f"Found {len(full)} images")  # Debug check

if len(full) > 0:
    train_ds, val_ds = random_split(full, [int(0.9*len(full)), len(full)-int(0.9*len(full))])
    bs = 32  # or your batch size
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
else:
    raise ValueError("No images found in the dataset!")