import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from dataset import *

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
  "dice":True
}

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
        if self.dice:
          return torch.argmax(x,axis=1)
        return x

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
        if self.dice:
          return torch.argmax(x,axis=1)
        return x





def get_loaders(batchSize, train_set, val_set, test_set):

  print("*** Create data loader ***")
  # Train
  train_loader_args = dict(batch_size=batchSize, num_workers=8)
  train_loader = DataLoader(train_set, **train_loader_args)
  
  if val_set == None and test_set==None:
    return train_loader
    
  # Dev

  val_loader_args = dict(batch_size=batchSize, num_workers=8)
  dev_loader = DataLoader(val_set, **val_loader_args)
  
  # Test
  test_loader_args = dict(batch_size=batchSize, num_workers=8)
  test_loader = DataLoader(test_set, **test_loader_args)
  
  return train_loader, dev_loader, test_loader

def evaluate(models, data_set, criterion, epoch, full=True):
  label_values = [2,3,4]
  start_time = time.time()
  total_loss = [0]*len(models)
  total_count = [0]*len(models)
  total_dice = [0]*len(models)
  sub_loss = [0]*len(models)
  sub_count = [0]*len(models)
  sub_dice = [0]*len(models)
  with torch.no_grad():
    for batch_idx, (original_data, target) in enumerate(data_set):
      original_data, original_target = original_data.cuda(), target.cuda()
      original_data = original_data.type(torch.cuda.FloatTensor)

      if not full:
        if batch_idx>100:
          break

      for i in range(len(models)):
        # target = (original_target>=label_values[i]).type(torch.cuda.FloatTensor)
        if i==0:
          target = (original_target>0).type(torch.cuda.FloatTensor)
        elif i==1:
          target = (original_target==4).type(torch.cuda.FloatTensor)
          target += (original_target==1).type(torch.cuda.FloatTensor)
        else:
          target = (original_target==1).type(torch.cuda.FloatTensor)
        model = models[i]
        # optimizer = optimizers[i]
        # optimizer.zero_grad()
        if i>0:
          data = original_data*output
        else: data = original_data
        output = model(data)        
        if hyper["dice"]:
          loss = criterion(output,target) 
        else:
          loss = criterion(output.type(torch.cuda.FloatTensor), target.type(torch.int64))   
        total_loss[i] += loss*original_data.shape[0]
        total_count[i] += original_data.shape[0]
        output=torch.argmax(output,axis=1)
        output=torch.unsqueeze(output,axis=1)     
        dice = Dice(output,target)
        total_dice[i] += dice*original_data.shape[0]
        sub_loss[i] += loss*original_data.shape[0]
        sub_count[i] += original_data.shape[0]
        sub_dice[i] += dice*original_data.shape[0]
        wandb.log({"val loss-{}".format(i): loss})

        if batch_idx>0 and batch_idx % 100 == 0:
            print("Evaluting Epoch: {}\tBatch: {}\tTimestamp: {}\tLoss: {}\tDice: {}".format(epoch, batch_idx, time.time() - start_time, sub_loss[i]/sub_count[i], sub_dice[i]/sub_count[i]))   
            wandb.log({"100 batch val loss-{}".format(i): sub_loss[i]/sub_count[i]})
            wandb.log({"100 batch val dice-{}".format(i): sub_dice[i]/sub_count[i]})
            sub_loss[i] = 0
            sub_count[i] = 0
            sub_dice[i] = 0 
      # if batch_idx > 2:  
      #   break

    for i in range(len(models)):
      total_loss[i] = total_loss[i]/total_count[i]
      total_dice[i] = total_dice[i]/total_count[i]
      print("Evaluting Epoch: {}\tTimestamp: {}\tLoss: {}\tDice: {}".format(epoch, time.time() - start_time, total_loss[i], total_dice[i])) 
      wandb.log({"val loss-{}".format(i): total_loss[i]})
      wandb.log({"val dice-{}".format(i): total_dice[i]})
    torch.cuda.empty_cache()

    return (total_loss, total_dice)

def train_epoch(models, data_set, criterion, optimizers, schedulers, epoch):
  start_time = time.time()
  label_values = [2,4,1]
  total_loss = [0]*len(models)
  total_count = [0]*len(models)
  total_dice = [0]*len(models)
  sub_loss = [0]*len(models)
  sub_count = [0]*len(models)
  sub_dice = [0]*len(models)
  for batch_idx, (original_data, original_target) in enumerate(data_set):
    # if batch_idx>=1:
    #   break

    original_data, original_target = original_data.cuda(), original_target.cuda()
    # data.requires_grad_(True)
    for i in range(len(models)):
      
      model = models[i]
      optimizer = optimizers[i]
      scheduler = schedulers[i]
      optimizer.zero_grad()
      data = original_data
      if i>0:
        data = original_data*torch.unsqueeze(target,axis=1)
        # data=torch.unsqueeze(data,axis=1)
      # data=torch.unsqueeze(data,axis=0)
      output = model(data)
      output = output.type(torch.cuda.FloatTensor)
      # target = (original_target>=label_values[i]).type(torch.cuda.FloatTensor)
      if i==0:
        target = (original_target>0).type(torch.cuda.FloatTensor)
      elif i==1:
        target = (original_target==4).type(torch.cuda.FloatTensor)
        target += (original_target==1).type(torch.cuda.FloatTensor)
      else:
        target = (original_target==1).type(torch.cuda.FloatTensor)
      if hyper["dice"]:
        output.requires_grad_(True)
        target.requires_grad_(True)
        loss = criterion(output,target) 
      else:
        loss = criterion(output.type(torch.cuda.FloatTensor), target.type(torch.int64))
      loss.backward()
      total_loss[i] += loss*original_data.shape[0]
      total_count[i] += original_data.shape[0]
      optimizer.step()
      output=torch.argmax(output,axis=1)
      output=torch.unsqueeze(output,axis=1)   
      dice = Dice(output,target)
      total_dice[i] += dice*original_data.shape[0]
      sub_loss[i] += loss*original_data.shape[0]
      sub_count[i] += original_data.shape[0]
      sub_dice[i] += dice*original_data.shape[0]

      wandb.log({"train loss-{}".format(i): loss})

      if batch_idx>0 and batch_idx % 100 == 0:
          print("Epoch: {}\t Model: {}\tBatch: {}\tTimestamp: {}\tLoss: {}\tDice: {}".format(epoch, i, batch_idx, time.time() - start_time, sub_loss[i]/sub_count[i], sub_dice[i]/sub_count[i]))  
          wandb.log({"100 batch train loss-{}".format(i): sub_loss[i]/sub_count[i]})
          wandb.log({"100 batch train dice-{}".format(i): sub_dice[i]/sub_count[i]})   
          schedulers[i].step(sub_dice[i]/sub_count[i])
          sub_loss[i] = 0
          sub_count[i] = 0
          sub_dice[i] = 0
          if i==2:
            evaluate(models, dev_loader ,criterion, epoch, full=False)
          
      # del loss
      # del data
      if i==2:
        del target
        del original_target
      torch.cuda.empty_cache()


    # if batch_idx > 1:
    #   break
  for i in range(len(models)):
    PATH = "./gdrive/MyDrive/11685/project/models/model-{}-epoch{}.w".format(i, epoch)
    print("Epoch completed")
    print("Training Epoch: {}\tTimestamp: {}\tLoss: {}\tDice: {}".format(epoch, time.time() - start_time, total_loss[i]/total_count[i], total_dice[i]/total_count[i]))   
    torch.save(models[i].state_dict(), PATH)  
    wandb.log({"train loss-{}".format(i): total_loss[i]/total_count[i]})
    wandb.log({"train dice-{}".format(i): total_dice[i]/total_count[i]}) 
  eval_loss, eval_dice = evaluate(models, dev_loader ,criterion, epoch)
  # for i in range(len(models)):
    
  


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
        del intersection
        del inputs
        del targets
        torch.cuda.empty_cache()
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
        del intersection
        del inputs
        del targets
        torch.cuda.empty_cache()
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



def main():

  #Create data loaders
  train_path = "data/Brats17TrainingData"
  val_path = "data/Brats17TrainingData"
  test_path = "data/Brats17TrainingData"

  train_set, val_set, test_set = Brats2017.split_dataset(train_path, n_samples=25)
  train_loader, dev_loader, test_loader = get_loaders(5, train_set, val_set, test_set)
  print(len(train_loader), len(dev_loader), len(test_loader))

  import wandb


  # # 1. Start a W&B run
  wandb.init(project="BrainCNN", entity="idl-gan-brain-tumors")
  wandb.config.update(hyper)

  np.random.seed(hyper["seed"])
  torch.manual_seed(hyper["seed"])
  torch.cuda.manual_seed(hyper["seed"])

  print("*** Create the model and define Loss and Optimizer ***")

  wnet = WNET(4, hyper["c_o"], hyper["c_l"])
  tnet = WNET(4,hyper["c_o"], hyper["c_l"])
  enet = ENET(4,hyper["c_o"], hyper["c_l"])

  #Kaiming init

  wnet, tnet, enet = wnet.cuda(), tnet.cuda(), enet.cuda()
  models = [wnet, tnet, enet]

  def weights_init(m):
      if isinstance(m, nn.Conv2d):
          torch.nn.init.kaiming_normal(m.weight.data)
          nn.init.constant(m.bias, 0)

  for model in models:  
    model.apply(weights_init)

  optimizer_w = optim.Adam(wnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
  optimizer_t = optim.Adam(tnet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
  optimizer_e = optim.Adam(enet.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])

  scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_w, 'max', factor=0.7, patience=3, threshold=0.01, verbose=True)
  scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'max', factor=0.7, patience=3, threshold=0.01, verbose=True)
  scheduler_e = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_e, 'max', factor=0.7, patience=3, threshold=0.01, verbose=True)


  optimizers = [optimizer_w, optimizer_t, optimizer_e]
  schedulers = [scheduler_w, scheduler_t, scheduler_e]


  if hyper["dice"]:
    criterion = DiceLoss()
  else:
    weight = torch.FloatTensor([0.005,0.995]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)

  # Train the model for N epochs
  for i in range(hyper["nEpochs"]):

    # Train
    print("Train\tEpoch: {}".format(i))
    startTime = time.time()
    train_epoch(models, train_loader, criterion, optimizers, schedulers, i)




