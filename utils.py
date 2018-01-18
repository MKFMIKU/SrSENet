import torch
import os

def save_checkpoint(model, rate, optim, epoch):
    model_folder = "SrSENet_x%d_checkpoints/"%rate
    model_out_path = model_folder + "{}.pth".format(epoch)

    state_dict = model.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))