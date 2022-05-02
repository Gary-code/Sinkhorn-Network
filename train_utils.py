import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os


def train_sorting_network(model, train_loader, loss, optim, device, epoch):
    """
    :param model:
    :param train_loader:
    :param loss:
    :param optim:
    :param device:
    :param epoch:
    :return:
    """
    model.train()
    loss_epoch = 0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(iter(train_loader))) as pbar:
        for it, (x_txt, x_vis, x_pos, gt_txt, gt_vis, gt_pos, hard_perm, _, _) in enumerate(iter(train_loader)):
            x_txt, x_vis, x_pos, gt_txt, gt_vis, gt_pos = x_txt.to(device), x_vis.to(device), x_pos.to(device), gt_txt.to(device), gt_vis.to(device), gt_pos.to(device)
            hard_perm = hard_perm.to(device)
            seq = torch.cat((x_txt, x_vis, x_pos), dim=-1)
            order_seq = torch.cat((gt_txt, gt_vis, gt_pos), dim=-1)
            out_seq = model(seq)  # [b, N, N]
            optim.zero_grad()
            inv_matrix = inv_soft_pers_flattened(out_seq)  # [b, N, N]
            local_loss = loss(hard_perm, inv_matrix)
            local_loss.backward()
            optim.step()
            loss_epoch += local_loss.item()
            pbar.set_postfix(loss_cap=loss_epoch/(it+1))
            pbar.update()
            # torch.cuda.empty_cache()
    print("Epoch {0:03d}: l2 loss={1:.4f}".format(epoch, local_loss / (len(train_loader))))
    return local_loss / (len(train_loader))


def inv_soft_pers_flattened(soft_perms_inf):
    n_number = soft_perms_inf.shape[-1]
    inv_soft_perms = torch.transpose(soft_perms_inf, 1, 2)

    inv_soft_perms_flat = inv_soft_perms.view(-1, n_number, n_number)
    return inv_soft_perms_flat



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



def eval_batch(model, data_loader, device, epoch):
    model.eval()
    equal_num = 0
    num_all = 0
    with torch.no_grad():
        acc_local_loss = 0
        for x_txt, x_vis, x_pos, _, _, _, hard_perm, det_ind, gt_ind in tqdm(data_loader, ascii=True, desc="val" + str(epoch)):
            x_txt, x_vis, x_pos, hard_perm = x_txt.to(device), x_vis.to(device), x_pos.to(device), hard_perm.to(device)
            det_ind, gt_ind = det_ind.to(device), gt_ind.to(device)
            seq = torch.cat((x_txt, x_vis, x_pos), dim=-1)
            soft_perm = inv_soft_pers_flattened(model(seq))
            loss = nn.MSELoss()
            local_loss = loss(hard_perm, soft_perm)
            acc_local_loss += local_loss.item()
            predict_matrix = F.one_hot(soft_perm.argmax(dim=-1), 3).float()
            pre_seq_ind = torch.matmul(predict_matrix, det_ind)
            num_all += len(det_ind)
            equal_num += (pre_seq_ind == gt_ind).all(dim=1).sum()
    acc = equal_num / num_all
    print(f"val {epoch} loss:", acc_local_loss / len(data_loader))
    print(f'val {epoch} acc:', acc)
    return acc_local_loss / len(data_loader), acc













