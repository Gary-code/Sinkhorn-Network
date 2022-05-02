import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from torch import nn
from data import dataset
from models import SinkhornNetwork
from train_utils import train_sorting_network, save_model, eval_batch
from torch.optim import Adam

print('training data preparing...')
train_loader = dataset.get_loader('train', fix_length=3, max_detections=10)
val_loader = dataset.get_loader('val', fix_length=3, max_detections=10)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

sinkhorn_net = SinkhornNetwork(3, 20, 0.6).to(device)

optim = Adam(sinkhorn_net.parameters(), lr=5e-4)

for epoch in range(0, 100):
    local_loss_avg = train_sorting_network(sinkhorn_net, train_loader, nn.MSELoss(), optim, device, epoch)
    acc_local_loss_avg, acc = eval_batch(sinkhorn_net, val_loader, device, epoch)
    save_model(sinkhorn_net, optim, epoch, -1, local_loss_avg, './save')