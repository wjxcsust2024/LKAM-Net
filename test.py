import torch
import numpy as np
from torch.utils.data import DataLoader
from Net8 import Net
import os
from dataloader2 import Datases_loader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 2 #3
model= Net().to(device)
model= model.eval() #315出问题新增
savedir = r'C:\Users\15504\Desktop\new4\weights\Net2_3.pth'
#imgdir = r'C:\Users\15504\Desktop\new6\1600\Data\Test\Image'
#labdir = r'C:\Users\15504\Desktop\new6\1600\Data\Test\Label'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack-  \CrackTree260\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\test_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_lab'
imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_img'
labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\SAR-sentinel\SAR-sentinel\test-img'
#labdir = r'C:\Users\15504\Desktop\new6\SAR-sentinel\SAR-sentinel\test-label'
#imgdir = r'C:\Users\15504\Desktop\new6\T-CRACK\test\images'
#labdir = r'C:\Users\15504\Desktop\new6\T-CRACK\test\labels'

imgsz = 512
resultsdir = r'C:\Users\15504\Desktop\new4\results\Net2_3'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)
    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img) #模型Net2的

        #o1,o2,o3,pred = model(img)#模型Net3的
        # print(pred.shape)
        # for i in pred:
        #     plt.imshow(i[0].cpu().detach().numpy())
        #     plt.show()
        #print(pred)
        #CE
        # B = pred.shape[0]
        # pred = pred.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        # pred = pred.argmax(1)
        # pred = pred.reshape(B, 1, imgsz, imgsz)
        #
        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy())

if __name__ == '__main__':
    test()