import torch, os, wandb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from accelerate import Accelerator
from unet import UNet
from dataset import ImageNoiseDataset
from utils import save_checkpoint, log_images


data_path = "data"
epochs = 50
bs = 16
lr = 1e-4
checkpoint_dir = "checkpoints"

wandb.init(project="unet-image-gen")
writer = SummaryWriter("runs/unet_image_gen")
accelerator = Accelerator()

full = ImageNoiseDataset(data_path)
train_ds, val_ds = random_split(full, [int(0.9*len(full)), len(full)-int(0.9*len(full))])
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=bs)


model = UNet()
opt = torch.optim.Adam(model.parameters(), lr=lr)

model, opt, train_dl, val_dl = accelerator.prepare(model, opt, train_dl, val_dl)

for epoch in range(epochs):
    model.train()
    for step, (noise, target) in enumerate(train_dl):
        pred = model(noise)
        loss = torch.nn.functional.mse_loss(pred, target)
        accelerator.backward(loss)
        opt.step()
        opt.zero_grad()

        if step % 10 == 0 and accelerator.is_main_process:
            wandb.log({"loss": loss.item()})
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_dl) + step)

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for noise, target in val_dl:
            pred = model(noise)
            val_loss += torch.nn.functional.mse_loss(pred, target).item()
        val_loss /= len(val_dl)

        if accelerator.is_main_process:
            wandb.log({"val_loss": val_loss})
            writer.add_scalar("Loss/val", val_loss, epoch)
            save_checkpoint(model, f"{checkpoint_dir}/unet_epoch_{epoch}.pt")
            log_images(noise[:4], pred[:4], target[:4], writer, epoch)

wandb.finish()