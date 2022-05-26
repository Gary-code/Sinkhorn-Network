import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from data import dataset
from models import SinkhornNetwork
from train_utils import inv_soft_pers_flattened


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('loading test data......')

test_loader = dataset.get_loader('test', 3, 10)

epo = 4
if not os.path.exists('result'):
    os.mkdir('result')

if not os.path.exists('score'):
    os.mkdir('score')


file_result = os.path.join('result', str(epo))
file_score = os.path.join('score', str(epo))


load_model = os.path.join('./save/', 'best_checkpoint.pth.tar')
classes = []

with open('./data/object_class_list.txt', 'r') as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())


classes.append("<pad>")

checkpoint = torch.load(load_model, map_location=torch.device("cpu"))

sinkhorn_net = SinkhornNetwork(3, 20, 0.6)

sinkhorn_net.load_state_dict(checkpoint['model_state_dict'])

sinkhorn_net = sinkhorn_net.to(device)

sinkhorn_net.eval()

equal_num = 0
num_all = 0
with torch.no_grad():
    acc_local_loss = 0
    for x_txt, x_vis, x_pos, _, _, _, hard_perm, det_ind, gt_ind in tqdm(test_loader, ascii=True, desc="test"):
        x_txt, x_vis, x_pos, hard_perm = x_txt.to(device), x_vis.to(device), x_pos.to(device), hard_perm.to(device)
        det_ind, gt_ind = det_ind.to(device), gt_ind.to(device)
        seq = torch.cat((x_txt, x_vis, x_pos), dim=-1)
        soft_perm = inv_soft_pers_flattened(sinkhorn_net(seq))
        loss = nn.MSELoss()
        local_loss = loss(hard_perm, soft_perm)
        acc_local_loss += local_loss.item()
        predict_matrix = F.one_hot(soft_perm.argmax(dim=-1), 3).float()
        pre_seq_ind = torch.matmul(predict_matrix, det_ind)
        f_result = open(file_result + '.txt', 'a')
        for (inp, pre, gt) in zip(det_ind, pre_seq_ind, gt_ind):
            f_result.write('real:' + str([classes[i.to(torch.int)] for i in inp]) + '\npredict:' + str([classes[i.to(torch.int)] for i in pre]) + '\ngt:' + str([classes[i.to(torch.int)] for i in gt]) + '\n\n')
        f_result.close()
        num_all += len(det_ind)
        equal_num += (pre_seq_ind == gt_ind).all(dim=1).sum()
        # true obj seq & pred obj seq
        references = []
        hypotheses = []
        idx2word = {index: word for word, index in enumerate(classes)}
        with open('result_seq_true.txt', 'w') as f:
            for r in tqdm(det_ind):
                words = [idx2word[i] for i in r]
                f.write(' '.join(words) + '\n')
                references.append([' '.join(words)])

        with open('result_seq_pred.txt', 'w') as f:
            for r in tqdm(gt_ind):
                words = [idx2word[i] for i in r]
                f.write(' '.join(words) + '\n')
                hypotheses.append([' '.join(words)])
        print(hypotheses[:5])
        print(references[:5])


acc = equal_num / num_all
f_score = open(file_score + '.txt', 'a')
f_score.write("test loss:" + str(acc_local_loss / len(test_loader)))
f_score.write('test acc:' + str(acc))
print(f"test loss:", acc_local_loss / len(test_loader))
print(f'test acc:', acc)
