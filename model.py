import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
 #hp here
epochs = 5
learn_rate = 0.1
batch= 3 

dataset = ImageFolder(root="./dataset/",transform=transform)
# print(dataset.class_to_idx) # check out labels
dataloaded = DataLoader(dataset,batch_size=batch,shuffle=False)

#amature video(batch) player 😅
# firstbatchimg,label = next(iter(dataload))
# plt.imshow(firstbatchimg[0].permute(1,2,0))
# plt.show(block=False)
# time.sleep(1)
# plt.close()
# plt.imshow(firstbatchimg[1].permute(1,2,0))
# plt.show(block=False)
# time.sleep(1)
# plt.close()
# plt.imshow(firstbatchimg[2].permute(1,2,0))
# plt.show(block=False)
# time.sleep(2)
# plt.close()

class Masknotes(nn.Module):
    def __init__(self):
        super(Masknotes,self).__init__()
        
        self.con1 = nn.Conv2d(9,18,5) # each chnnl two kernel
        self.pool= nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(18,36,5)
        self.l1 = nn.Linear(133956 ,120)
        self.l2 = nn.Linear(120,84)
        self.l3 = nn.Linear(84,2)
    def forward(self,x):
        x = self.pool(F.relu(self.con1(x)))
        x = self.pool(F.relu(self.con2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x)) 
        x = self.l3(x)

        return x
    
model = Masknotes().to(device) # just send to gpu mem

loss_metric = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate)

count_epoch=0 

# it club up 3 img so that 9 channels    
batch_club = []  
label_club = []  

for batch_num, (imgs, labels) in enumerate(dataloaded):
    imgs, labels = imgs.to(device), labels.to(device)

    batch_club.append(imgs)  
    label_club.append(labels)

    if len(batch_club) == 3:  
        stacked_imgs = torch.cat(batch_club, dim=1)  
        stacked_labels = label_club[0] 
        stacked_imgs = stacked_imgs.view(-1, 9, 256, 256)

        # forward
        output = model(stacked_imgs)
        loss = loss_metric(output, stacked_labels)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count_epoch+=1
        print("\nEpoch:", count_epoch + 1,"\nLoss: ", round(loss.item(),5))
        batch_club = []
        label_club = []
    if count_epoch==epochs:
        break