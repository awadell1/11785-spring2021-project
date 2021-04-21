import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from dataset import *


#modified code from: 


class conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )
    def forward(self, x):
        x=self.conv(x)
        return x
class inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inconv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )
    def forward(self, x):
        x=self.conv(x)
        return x
class res_block(nn.Module):
    ''''''
    def __init__(self,in_ch,out_ch,d=1):
        super(res_block,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        x1 = self.doubleconv(x)
        x =x+ x1
        return x
class anistropic_conv(nn.Module):
    '''1X1X3'''
    def __init__(self,in_ch,out_ch):
        super(anistropic_conv,self).__init__()
        self.aniconv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,(1,1,3),padding=(0,0,1),dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )
    def forward(self,x):
        x1=self.aniconv(x)
        return x1
class Block_3(nn.Module):
    def __init__(self,in_ch,out_ch,flag):
        super(Block_3,self).__init__()
        if flag==1:
            self.block = nn.Sequential(
                res_block(in_ch, out_ch, 1),
                res_block(out_ch, out_ch, 2),
                res_block(out_ch, out_ch, 3),
                anistropic_conv(out_ch, out_ch)
            )
        else:
            self.block = nn.Sequential(
                res_block(in_ch, out_ch, 3),
                res_block(out_ch, out_ch, 2),
                res_block(out_ch, out_ch, 1),
                anistropic_conv(out_ch, out_ch)
            )
    def forward(self, x):
        x1=self.block(x)
        return x1
class Block_2(nn.Module):
    def __init__(self,in_ch,out_ch,flag):
        super(Block_2,self).__init__()
        self.flag=flag
        self.block=nn.Sequential(
            res_block(in_ch,out_ch),
            res_block(out_ch,out_ch),
            anistropic_conv(out_ch,out_ch),
        )
        self.pooling=nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,1), stride=(2,2,1),padding=(1,1,0)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x1=self.block(x)
        out=self.pooling(x1)
        if self.flag==1:
            return x1,out
        else:
            return out
class up(nn.Module):
    def __init__(self,in_ch,out_classes,flag):
        super(up,self).__init__()
        if flag==2:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0)),
                nn.ConvTranspose3d(out_classes, out_classes, kernel_size=(3,3,1), \
                           stride=(2,2,1),padding=(1,1,0),output_padding=(1,1,0)),
                nn.BatchNorm3d(out_classes),
                nn.PReLU(),
            )
        if flag==4:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0)),
                nn.ConvTranspose3d(out_classes, out_classes, kernel_size=(3,3,1), \
                           stride=(2,2,1),padding=(1,1,0),output_padding=(1,1,0)),
                nn.BatchNorm3d(out_classes),
                nn.PReLU(),
                nn.ConvTranspose3d(out_classes,out_classes, kernel_size=(3,3,1), \
                           stride=(2,2,1),padding=(1,1,0),output_padding=(1,1,0)),
                nn.BatchNorm3d(out_classes),
                nn.PReLU(),
            )
        if flag==1:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0)),
            )


    def forward(self, x):
        x=self.conv(x)
        return x
class WNET(nn.Module):
    def __init__(self,n_channels,out_ch,n_classes, dice=hyper["dice"]):
        super(WNET,self).__init__()
        self.conv=conv(n_channels,out_ch)
        self.block0=Block_2(out_ch,out_ch,0)
        self.block1=Block_2(out_ch,out_ch,1)
        self.block2 = Block_3(out_ch, out_ch,1)
        self.block3 = Block_3(out_ch, out_ch, 0)
        self.up0=up(out_ch,n_classes,2)
        self.up1 = up(out_ch, n_classes*2, 4)
        self.up2 = up(out_ch, n_classes*4, 4)
        self.out=up(7*n_classes,n_classes,1)
        self.dice = dice

    def forward(self, x):
        x=self.conv(x)
        x=self.block0(x)
        x0,x=self.block1(x)
        x0=self.up0(x0)
        x=self.block2(x)
        x1=self.up1(x)
        x=self.block3(x)
        x=self.up2(x)
        x=torch.cat([x0,x1,x],dim=1)
        x=self.out(x)
        x= F.sigmoid(x)
        # if self.dice:
        #   return x, torch.argmax(x,axis=1)
        return x, torch.argmax(x,axis=1)

class ENET(nn.Module):
    def __init__(self,n_channels,out_ch,n_classes,dice=hyper["dice"]):
        super(ENET, self).__init__()
        self.conv = conv(n_channels, out_ch)
        self.block0 = Block_2(out_ch, out_ch, 1)
        self.block1 = Block_2(out_ch, out_ch, 1)
        self.block2 = Block_3(out_ch, out_ch, 1)
        self.block3 = Block_3(out_ch, out_ch, 0)
        self.up0 = up(out_ch, n_classes, 1)
        self.up1 = up(out_ch, n_classes * 2, 2)
        self.up2 = up(out_ch, n_classes * 2, 2)
        self.out = up(5 * n_classes, n_classes, 1)
        self.dice = dice

    def forward(self, x):
        x = self.conv(x)
        x,_ = self.block0(x)
        x0, x = self.block1(x)
        x0 = self.up0(x0)
        x = self.block2(x)
        x1 = self.up1(x)
        x = self.block3(x)
        x = self.up2(x)
        x = torch.cat([x0, x1, x], dim=1)
        x = self.out(x)
        x= F.sigmoid(x)
        return x, torch.argmax(x,axis=1)


def get_loaders(batchSize, train_set, val_set, test_set):

  print("*** Create data loader ***")
  # Train
  train_loader_args = dict(shuffle=True, batch_size=batchSize, num_workers=8)
  train_loader = DataLoader(train_set, **train_loader_args)
  
  if val_set == None and test_set==None:
    return train_loader
    
  # Dev
  dev_loader = DataLoader(val_set, **train_loader_args)
  
  # Test
  test_loader_args = dict(shuffle=False, batch_size=batchSize, num_workers=8)
  test_loader = DataLoader(test_set, **test_loader_args)
  
  return train_loader, dev_loader, test_loader

def evaluate(models, data_set, criterion, epoch):
  label_values = [1,2,4]
  start_time = time.time()
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(data_set):
      data, original_target = data.cuda(), target.cuda()
      for i in range(len(models)):
        target = (original_target==label_values[i]).type(torch.cuda.FloatTensor)
        data=data.type(torch.cuda.FloatTensor)
        model = models[i]
        if i>0:
          data=torch.unsqueeze(data,axis=1)
        # optimizer = optimizers[i]
        # optimizer.zero_grad()
        if hyper["dice"]:
          out, data = model(data)
        else:
          out, data = model(data)        
        if hyper["dice"]:
          loss = criterion(data,target) 
        else:
          loss = criterion(out.type(torch.cuda.FloatTensor), target.type(torch.int64))        
      dice = Dice(data,target)
      if batch_idx % 50 == 0:
          print("Evaluating Epoch: {}\tBatch: {}\tTimestamp: {}\tLoss: {}\tDice:{}".format(epoch, batch_idx, time.time() - start_time, loss, dice))   
      # if batch_idx > 2:  
      #   break
      torch.cuda.empty_cache()


def train_epoch(models, data_set, criterion, optimizers, epoch):
  start_time = time.time()
  label_values = [1,2,4]

  for batch_idx, (data, target) in enumerate(data_set):
    # if batch_idx>=0:
    #   break
    data, original_target = data.cuda(), target.cuda()
    # data.requires_grad_(True)
    for i in range(len(models)):
      
      model = models[i]
      optimizer = optimizers[i]
      optimizer.zero_grad()
      if i>0:
        data = target
        data=torch.unsqueeze(data,axis=1)
      # data=torch.unsqueeze(data,axis=0)
      if hyper["dice"]:
        out, data = model(data)
      else:
        out, data = model(data)
      data = data.type(torch.cuda.FloatTensor)
      target = (original_target==label_values[i]).type(torch.cuda.FloatTensor)
      data.requires_grad_(True)
      target.requires_grad_(True)
      if hyper["dice"]:
        loss = criterion(data,target) 
      else:
        loss = criterion(out.type(torch.cuda.FloatTensor), target.type(torch.int64))
      loss.backward()
      optimizer.step()
      if batch_idx % 50 == 0:
          print("Epoch: {}\tBatch: {}\tTimestamp: {}\tLoss: {}".format(epoch, batch_idx, time.time() - start_time, loss))     
      torch.cuda.empty_cache()
    # if batch_idx > 1:
    #   break
    
  evaluate(models, dev_loader ,criterion, epoch)


class DiceLoss(nn.Module):
  #Code: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=2):

        # assert(inputs.shape==targets.shape)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)


        #Getting weirdly large sum of input. Predicting stuff as 1.
        # print("Sum of input:{}, targets:{}".format(inputs.sum(), targets.sum()))
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def Dice(inputs, targets, smooth=2):

  # assert(inputs.shape==targets.shape)
  
  #comment out if your model contains a sigmoid or equivalent activation layer
  
  #flatten label and prediction tensors
  inputs = inputs.view(-1)
  targets = targets.view(-1)

  #Getting weirdly large sum of input. Predicting stuff as 1.
  # print("Sum of input:{}, targets:{}".format(inputs.sum(), targets.sum()))
  
  intersection = (inputs * targets).sum()                            
  dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
  
  return dice


hyper = {
  "dataPath": "./data_root",
  "batchSize": 5,
  "lr":1e-3,
  "weightDecay":1e-7,
  "nEpochs":20,
  "checkpointPath": "./gdrive/MyDrive/11685/project/checkpoint/",
  "seed":20,
  "load_model":False,
  "c_o":32,
  "c_l":2,
  "dice":False
}

def main():
    #Create data loaders
    train_path = "data/Brats17TrainingData"
    val_path = "data/Brats17TrainingData"
    test_path = "data/Brats17TrainingData"

    dataset = Brats2017(train_path)
    train_set, val_set, test_set = dataset.split_dataset(train_path)
    train_loader, dev_loader, test_loader = get_loaders(5, train_set, val_set, test_set)

    np.random.seed(hyper["seed"])
    torch.manual_seed(hyper["seed"])
    torch.cuda.manual_seed(hyper["seed"])

    print("*** Create the model and define Loss and Optimizer ***")

    wnet = WNET(4, hyper["c_o"], hyper["c_l"])
    tnet = WNET(1,hyper["c_o"], hyper["c_l"])
    enet = ENET(1,hyper["c_o"], hyper["c_l"])

    wnet, tnet, enet = wnet.cuda(), tnet.cuda(), enet.cuda()
    models = [wnet, tnet, enet]

    # checkpoint = torch.load(hyper["savedCheckpoint"])
    # model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_w = optim.Adam(wnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
    optimizer_t = optim.Adam(tnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
    optimizer_e = optim.Adam(enet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])

    optimizers = [optimizer_w, optimizer_t, optimizer_e]

    if hyper["dice"]:
        criterion = DiceLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Train the model for N epochs
    for i in range(hyper["nEpochs"]):

        # Train
        print("Train\tEpoch: {}".format(i))
        startTime = time.time()
        train_epoch(models, train_loader, criterion, optimizers, i)

