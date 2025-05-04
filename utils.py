import torch
import torchvision
import os
import glob



def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
    checkpoint_dir = os.path.dirname(path)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pt')))
    
    if len(checkpoint_files) > 10:
        for old_checkpoint in checkpoint_files[:-10]:
            os.remove(old_checkpoint)

            

def log_images(noise, output, target, writer, step):
    grid = torchvision.utils.make_grid(torch.cat([noise, output, target], dim=0), nrow=noise.size(0))
    writer.add_image("Samples", grid, step)