import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from Net8 import Net
from dataloader1 import Datases_loader as dataloader
from Dice_BCEwithLogits import SoftDiceLoss as bcedice
def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.9, max_iter=80):
    return initial_lr * ((1 - step / float(max_iter)) ** power)
def adjust_lr(optimizer, step, initial_lr):
    lr = inverseDecayScheduler(step, initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batchsz = 4
lr = 0.0005
items = 40
model = Net().to(device)
criterion = bcedice()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)

savedir = r'C:\Users\15504\Desktop\new4\weights\Net2_3.pth'
imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\train_img'
labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\train_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\train_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\train_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\train_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\train_lab'


imgsz = 512
dataset = dataloader(imgdir, labdir, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)

lossx = 0
ls_loss = []
def train():
    for epoch in range(items):
        lossx = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()
            pred = model(img)  # 模型Net2的
            b, _, h, w = lab.shape

            loss = criterion(pred, lab)

            loss.backward()
            optimizer.step()

            lossx = lossx + loss

        adjust_lr(optimizer, epoch, optimizer.param_groups[0]['lr'])
        lossx = lossx / dataset.num_of_samples()
        ls_loss.append(lossx.item())
        print('epoch' + str(epoch + 1) + '---loss:' + str(lossx.item()))

    torch.save(model.state_dict(), savedir)


if __name__ == '__main__':
    train()

    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(F1)
    print(ls_loss)
    str = 'loss = ' + str(ls_loss)
    # str = 'accuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(F1) + '\nloss:' + str(ls_loss)
    filename = r'C:\Users\15504\Desktop\new4\weights\Net2_3.txt'
    with open(filename, mode='w', newline='') as f:
        f.writelines(str)
