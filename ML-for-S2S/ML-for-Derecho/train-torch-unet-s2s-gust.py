import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import matplotlib.pyplot as plt
import xskillscore as xs

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torch_funcs
import torch_s2s_dataset_gust

############################################################################
############################################################################

parser = argparse.ArgumentParser(description='Training UNET for bias-correcting a field (tas2m/prsfc) from CESM.')

parser.add_argument("--variable",choices=["tas2m", "prsfc"],required=True,type=str, help="Variable to be corrected.")
parser.add_argument("--week",choices=["1", "2", "3","4","5","6"],required=True,type=str, help="Week (lead time) predicted.")
parser.add_argument("--data_dir",required=False,default='/glade/gust/scratch/molina/', type=str, help="Data storage location.")
parser.add_argument("--dates_training",required=False,default="1999-02-01/2015-12-31", type=str, help="Initial and final dates of training subset with the format '1999-02-01/2015-12-31'")
parser.add_argument("--dates_val",required=False,default="2016-01-01/2017-12-31", type=str, help="Initial and final dates of validation subset with the format '2016-01-01/2017-12-31'")
parser.add_argument("--dates_test",required=False,default="2018-01-01/2020-12-31", type=str, help="Initial and final dates of test subset with the format '2018-01-01/2020-12-31'")
parser.add_argument("--region_type",choices=["fixed", "random"],default='fixed',required=False, type=str,help="Region method used. Defaults 'fixed' uses one region. 'random' changes regions.")

parser.add_argument("--lon0",required=True, type=str, help="Bottom left corner of 'fixed' region (0 to 360).")
parser.add_argument("--lat0",required=True, type=str, help="Bottom left corner of 'fixed' region (-90 to 90).")
parser.add_argument("--dxdy",required=False, type=str, default="32", help="number of grid cells for 'fixed' or 'random' region. Defaults to 32.")


#### HYPERPARAMETERS ####

parser.add_argument("--k1",required=False, type=str, default="3", help="Kernel size 1")
parser.add_argument("--p1",required=False, type=str, default="1", help="Padding size 1")
parser.add_argument("--k2",required=False, type=str, default="3", help="Kernel size 2")
parser.add_argument("--p2",required=False, type=str, default="1", help="Padding size 2")

parser.add_argument("--batch_size",required=False, type=str, default="64", help="Batch size.")
parser.add_argument("--learning_rate",required=False, type=str, default="1e-3", help="Learning rate.")
parser.add_argument("--num_epochs",required=False, type=str, default="30", help="Number of epochs.")

args=parser.parse_args()

var = args.variable
wks = int(args.week)
data_dir = args.data_dir
dates_train = args.dates_training.split('/')
dates_val = args.dates_val.split('/')
dates_test = args.dates_test.split('/')
region_type = args.region_type

lat0 = float(args.lat0)
lon0 = float(args.lon0)
dxdy = float(args.dxdy)

BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
NUM_EPOCHS = int(args.num_epochs)

k1=int(args.k1)
p1=int(args.p1)
k2=int(args.k2)
p2=int(args.p2)

# print (var,wks,data_dir)
# print(dates_train)
# print(dates_val)
# print(dates_test)
# print(lat0,lon0,dxdy)
# print(BATCH_SIZE,LEARNING_RATE,k1,p1,k2,p2,NUM_EPOCHS)
# aaaaa
############################################################################
############################################################################

def reverse_negone(ds, minv, maxv):
    """
    reversing negative zero-to-one normalization
    """
    return (((ds + 1) / 2) * (maxv - minv)) + minv

train = torch_s2s_dataset_gust.S2SDataset(
    
    week=wks, variable=var, norm='minmax', region=region_type,
    
    minv=None, maxv=None, mnv=None, stdv=None,
    
    lon0=lon0, lat0=lat0, dxdy=dxdy, feat_topo=True, feat_lats=True, feat_lons=True,
    
    startdt=dates_train[0], enddt=dates_train[1], homedir=data_dir
)

valid = torch_s2s_dataset_gust.S2SDataset(
    
    week=wks, variable=var, norm='minmax', region=region_type,
    
    minv=train.min_val, maxv=train.max_val, mnv=None, stdv=None,
    
    lon0=lon0, lat0=lat0, dxdy=dxdy, feat_topo=True, feat_lats=True, feat_lons=True,
    
    startdt=dates_val[0], enddt=dates_val[1], homedir=data_dir
)

tests = torch_s2s_dataset_gust.S2SDataset(
    
    week=wks, variable=var, norm='minmax', region=region_type,
    
    minv=train.min_val, maxv=train.max_val, mnv=None, stdv=None,
    
    lon0=lon0, lat0=lat0, dxdy=dxdy, feat_topo=True, feat_lats=True, feat_lons=True,
    
    startdt=dates_test[0], enddt=dates_test[1], homedir=data_dir
)


train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
tests_loader = DataLoader(tests, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


def train_func(model, dataloader, nc):
    """
    Training function.
    
    Args:
        model (torch): pytorch neural network.
        dataloader (torch): pytorch dataloader.
        nc (int): number of channels.
    """
    model.train()
    
    running_loss = 0.0
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    
    for data in dataloader:
        
        img_noisy = data['input'].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        img_label = data['label'].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(img_noisy)
        
        loss = criterion(outputs, img_label)
        closs = torch_funcs.corrcoef(outputs, img_label) # corr b/w unet and era5
        tloss = torch_funcs.corrcoef(img_noisy[:,nc-1:nc,:,:], img_label) # corr b/w cesm2 and era5
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        corrcoef_loss += closs.item()
        corrcoef_true += tloss.item()
    
    train_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)
    
    return train_loss, coef_loss, coef_true

def valid_func(model, dataloader, nc):
    """
    Validation function.
    
    Args:
        model: pytorch neural network.
        dataloader: pytorch dataloader.
        nc (int): number of channels.
    """
    model.eval()
    
    running_loss = 0.0
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    
    with torch.no_grad():
        
        for i, data in enumerate(dataloader):

            img_noisy = data['input'].squeeze(dim=2)
            img_noisy = img_noisy.to(device, dtype=torch.float)

            img_label = data['label'].squeeze(dim=2)
            img_label = img_label.to(device, dtype=torch.float)

            outputs = model(img_noisy)
            
            loss = criterion(outputs, img_label)
            closs = torch_funcs.corrcoef(outputs, img_label) # corr b/w unet and era5
            tloss = torch_funcs.corrcoef(img_noisy[:,nc-1:nc,:,:], img_label) # corr b/w cesm2 and era5
            
            running_loss += loss.item()
            corrcoef_loss += closs.item()
            corrcoef_true += tloss.item()

    val_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)
    
    return val_loss, coef_loss, coef_true


def get_device():
    """
    Grab GPU (cuda).
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
        
    else:
        device = 'cpu'
        
    return device


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k1, p1, k2, p2, mid_channels=None):
        
        super().__init__()
        
        if not mid_channels:
            
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=k2, padding=p2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, k1, p1, k2, p2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, k1, p1, k2, p2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, k1, p1, k2, p2, bilinear=True):
        
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        
        if bilinear:
            
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, k1, p1, k2, p2, mid_channels=in_channels // 2)
            
        else:
            
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, k1, p1, k2, p2)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        return self.conv(x)
    
    
class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, k1=3, p1=1, k2=3, p2=1, bilinear=True, mask=None):
        
        super(UNet, self).__init__()
        
        factor = 2 if bilinear else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mask = mask

        self.inc = DoubleConv(n_channels, 32 * 2, k1, p1, k2, p2)
        
        self.down1 = Down(32 * 2, 64 * 2, k1, p1, k2, p2)
        self.down2 = Down(64 * 2, 128 * 2, k1, p1, k2, p2)
        self.down3 = Down(128 * 2, 256 * 2, k1, p1, k2, p2)
        self.down4 = Down(256 * 2, 512 * 2 // factor, k1, p1, k2, p2)
        
        self.up1 = Up(512 * 2, 256 * 2 // factor, k1, p1, k2, p2, bilinear)
        self.up2 = Up(256 * 2, 128 * 2 // factor, k1, p1, k2, p2, bilinear)
        self.up3 = Up(128 * 2, 64 * 2 // factor, k1, p1, k2, p2, bilinear)
        self.up4 = Up(64 * 2, 32 * 2, k1, p1, k2, p2, bilinear)
        
        self.outc = OutConv(32 * 2, n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return logits
    
    
nc = 4  # terrain height (era5), lats, lons, raw temperature
net = UNet(n_channels=nc, n_classes=1, k1=k1, p1=p1, k2=k2, p2=p2, bilinear=True)

# the optimizer
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# the loss function
#criterion = nn.MSELoss()   # adding weights to positive class
#criterion = nn.L1Loss()
criterion = torch.nn.SmoothL1Loss()


device = get_device()
print(device)
net.to(device)

train_loss = []
valid_loss = []

train_corr = []
valid_corr = []

train_true = []
valid_true = []

for enum, epoch in enumerate(range(NUM_EPOCHS)):
    
    t_loss, t_corr, t_true = train_func(net, train_loader, nc)
    v_loss, v_corr, v_true = valid_func(net, valid_loader, nc)
    
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
    
    train_corr.append(t_corr)
    valid_corr.append(v_corr)
    
    train_true.append(t_true)
    valid_true.append(v_true)
    
    print(f"Epoch {epoch + 1} of {NUM_EPOCHS}; "\
          f"Train: {t_loss:.4f}, {t_corr:.4f}, {t_true:.4f}; "\
          f"Val: {v_loss:.4f}, {v_corr:.4f}, {v_true:.4f}")