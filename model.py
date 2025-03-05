import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda")
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



class Masknotes(nn.Module):
    def __init__(self):
        super(Masknotes,self).__init__()
        
        self.model1=models.resnet18(pretrained=True)
self.model2= models.resnet18(pretrained=True)
self.model3= models.resnet18(pretrained=True)
self.l1 = nn.Linear(512+512+512,64) 
self.l2 = nn.Linear(64,20)
self.l3 = nn.Linear(10,2)
    def forward(self,x1,x2,x3):
        x = self.model1(x1)
        x = self.model2(x2)
        x = self.model3(x3)
        x = torch.cat((x1,x2,x3), start_dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x)) 
        x = self.l3(x)

        return x
    
model = Masknotes().to(device) # just send to gpu mem

loss_metric = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate)

count_epoch = 0
batch_club = []  # clubing three img
label_club = []  

for batch_num, (imgs, labels) in enumerate(dataloaded):
    imgs, labels = imgs.to(device), labels.to(device)

    batch_club.append(imgs)  
    label_club.append(labels) 

    if len(batch_club) == 3:  
        img1, img2, img3 = batch_club    
        stacked_labels = label_club[0]  

        #forward
        output = model(img1, img2, img3)
        loss = loss_metric(output, stacked_labels)

        #optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count_epoch += 1
        print(f"\nEpoch: {count_epoch}\nLoss: {round(loss.item(), 5)}")

        
        batch_club = []
        label_club = []

    #epoch break
    if count_epoch == epochs:
        break