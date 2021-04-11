import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from dataset import *


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
        self.inconv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )
    def forward(self, x):
        x=self.inconv(x)
        return x
class res_block(nn.Module):
    def __init__(self,in_ch,out_ch,d=1):
        super(res_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        x1 = self.conv(x)
        x =x+x1
        return x
class anistropic_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(anistropic_conv,self).__init__()
        self.anistropic_conv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,(1,1,3),padding=(0,0,1),dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )
    def forward(self,x):
        x1=self.anistropic_conv(x)
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
    def __init__(self,n_channels,out_ch,n_classes):
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
        return F.sigmoid(x)
class ENET(nn.Module):
    def __init__(self,n_channels,out_ch,n_classes):
        super(ENET, self).__init__()
        self.conv = conv(n_channels, out_ch)
        self.block0 = Block_2(out_ch, out_ch, 1)
        self.block1 = Block_2(out_ch, out_ch, 1)
        self.block2 = Block_3(out_ch, out_ch, 1)
        self.block3 = Block_3(out_ch, out_ch, 0)
        self.up1 = up(out_ch, n_classes * 2, 2)
        self.up2 = up(out_ch, n_classes * 2, 2)
        self.out = up(5 * n_classes, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x,_ = self.block0(x)
        x0, x = self.block1(x)
        x = self.block2(x)
        x1 = self.up1(x)
        x = self.block3(x)
        x = self.up2(x)
        x = torch.cat([x0, x1, x], dim=1)
        x = self.out(x)
        return F.sigmoid(x)

def get_loaders(batchSize, train, val=None, test=None):

  print("*** Create data loader ***")
  # Train
  train_loader_args = dict(shuffle=True, batch_size=batchSize, num_workers=8)
  train_loader = DataLoader(Brats2017(train), **train_loader_args)
  
  if val == None and test==None:
    return train_loader
    
  # Dev
  dev_loader = DataLoader(Brats2017(val), **train_loader_args)
  
  # Test
  test_loader_args = dict(shuffle=False, batch_size=batchSize, num_workers=8)
  test_loader = DataLoader(Brats2017(test), **test_loader_args)
  
  return train_loader, dev_loader, test_loader

def train_epoch(models, data_set, criterion, optimizers, epoch):
  start_time = time.time()
  for batch_idx, (data, target) in enumerate(data_set):
    data, target = data.cuda(), target.cuda()
    #since we are using dataset for now
    data =torch.unsqueeze(data,0)
    #[TODO] target needs to extract binding box
    for i in range(len(models)):
      model = models[i]
      optimizer = optimizers[i]
      optimizer.zero_grad()
      data = model(data)
      #[TODO] output needs to extract binding box
      loss = criterion(data,target) 
      loss.backward()
      optimizer.step()
      if batch_idx % 50 == 0:
          print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time() - start_time))     
      torch.cuda.empty_cache()
      del data
      del target


class DiceLoss(nn.Module):
  #Code: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

hyper = {
  "dataPath": "./data_root",
  "batchSize": 1,
  "lr":1e-3,
  "weightDecay":1e-7,
  "nEpochs":20,
  "checkpointPath": "./gdrive/MyDrive/11685/project/checkpoint/",
  "seed":20,
  "load_model":False,
  "c_o":32,
  "c_l":2
}

def main():
  np.random.seed(hyper["seed"])
  torch.manual_seed(hyper["seed"])
  torch.cuda.manual_seed(hyper["seed"])

  print("*** Create the model and define Loss and Optimizer ***")

  wnet = WNET(144, hyper["c_o"], hyper["c_l"])
  tnet = WNET(hyper["c_o"],hyper["c_o"], hyper["c_l"])
  enet = ENET(hyper["c_o"],hyper["c_o"], hyper["c_l"])

  wnet, tnet, enet = wnet.cuda(), tnet.cuda(), enet.cuda()
  models = [wnet, tnet, enet]

  # checkpoint = torch.load(hyper["savedCheckpoint"])
  # model.load_state_dict(checkpoint["model_state_dict"])
  optimizer_w = optim.Adam(wnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
  optimizer_t = optim.Adam(tnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
  optimizer_e = optim.Adam(enet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])

  optimizers = [optimizer_w, optimizer_t, optimizer_e]

  criterion = DiceLoss()

  # Train the model for N epochs
  for i in range(hyper["nEpochs"]):

    # Train
    print("Train\tEpoch: {}".format(i))
    startTime = time.time()
    train_epoch(models, train_set, criterion, optimizers, i)
