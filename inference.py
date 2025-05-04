import torch
from torchvision.utils import save_image
from unet import UNet

model = UNet()
model.load_state_dict(torch.load("checkpoints/unet_epoch_4.pt"))
model.eval()

noise = torch.randn(4, 3, 256, 256)
with torch.no_grad():
    output = model(noise)

save_image(output, "samples/generated.png")