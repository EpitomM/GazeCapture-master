import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    # 左眼、右眼和从原始帧中检测和裁剪出来的脸部图像 尺寸均为 224×224
    # 人脸网格（face grid） 尺寸为 25×25
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            # Conv2d：output = ((input_size - kernel_size + 2 * padding) / stride ) + 1 向下取整
            # MaxPool2d：(input_size - kernel_size) / stride + 1  向下取整
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),              # (224, 224, 3) -> (54, 54, 96)     (224-11+2*0)/4+1=54     CONV-E1, CONV-F1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                              # (54, 54, 96) -> (26, 26, 96)      (54-3)/2+1=26
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),   # (26, 26, 96) -> (26, 26, 256)     (26-5+2*2)/1+1=26       CONV-E2, CONV-F2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                              # (26, 26, 256) -> (12, 12, 256)    (26-3)/2+1=12
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),            # (12, 12, 256) -> (12, 12, 384)    (12-3+2*1)/1+1=12       CONV-E3, CONV-F3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),             # (12, 12, 384) -> (12, 12, 64)       (12-1)/1+1=12         CONV-E4, CONV-F4
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)                                               # 12 * 12 * 64
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),           # FC-F1
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),                 # FC-F2
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),    # FC-FG1
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),                    # FC-FG2
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ITrackerModel(nn.Module):


    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*12*12*64, 128),     # FC-E1
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 128),     # FC1
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),              # FC2
            )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x
