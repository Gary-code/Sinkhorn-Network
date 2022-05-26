import os
import torch


def save_model(model, model_optim, epoch, it, local_loss, save_folder):

    dir_path = save_folder
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    PATH = os.path.join(save_folder, str(epoch) + '_' + str(it) + '.tar')

    checkpoint = {
        'epoch': epoch,
        'iter': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model_optim.state_dict(),
        'local_loss': local_loss
    }

    torch.save(checkpoint, PATH)


def save_best_checkpoint(model, model_optim, epoch, it, local_loss, save_folder):
    dir_path = save_folder
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    PATH = os.path.join(save_folder, 'best_checkpoint.pth.tar')

    checkpoint = {
        'epoch': epoch,
        'iter': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model_optim.state_dict(),
        'local_loss': local_loss
    }

    torch.save(checkpoint, PATH)


def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
