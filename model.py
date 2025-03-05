import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
device = torch.device("cuda")
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
 #hp here
epochs = 25
learn_rate = 0.001
batch= 1 

dataset = ImageFolder(root="./dataset/",transform=transform)
# print(dataset.class_to_idx) # check out labels
dataloaded = DataLoader(dataset,batch_size=batch,shuffle=False)



class Masknotes(nn.Module):
    def __init__(self):
        super(Masknotes,self).__init__()
        
        self.model1=models.resnet18(pretrained=True)
        self.model2= models.resnet18(pretrained=True)
        self.model3= models.resnet18(pretrained=True)
        self.model1 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.model2 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.model3 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.l1 = nn.Linear(512+512+512,64) 
        self.l2 = nn.Linear(64,20)
        self.l3 = nn.Linear(20,2)
    def forward(self,x1,x2,x3):
        
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model3(x3)
        x1 = torch.flatten(x1, start_dim=1) 
        x2 = torch.flatten(x2, start_dim=1)
        x3 = torch.flatten(x3, start_dim=1)

        x = torch.cat((x1,x2,x3), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x)) 
        x = self.l3(x)

        return x
    
model = Masknotes().to(device) # just send to gpu mem

if os.path.exists("model.pth"): # loading model
    model.load_state_dict(torch.load("model.pth"), strict=False)
    print("model loaded")

loss_metric = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate)

count_epoch = 0
img_club =[]
for batch_num, (imgs, labels) in enumerate(dataloaded):
    imgs, labels = imgs.to(device), labels.to(device)
    img_club.append(imgs)
    if len(img_club)==3:
        # forward
        output = model(img_club[0],img_club[1],img_club[2])
        loss = loss_metric(output, labels)
         # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        img_club =[]
        count_epoch += 1
        print(f"\nEpoch: {count_epoch}\nLoss: {round(loss.item(), 5)}")
        
        if count_epoch == epochs:
            break
 
torch.save(model.state_dict(), "model.pth")
