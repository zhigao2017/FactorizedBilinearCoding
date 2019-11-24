#coding:utf-8
from torch import nn
import torch
#from .BasicModule import BasicModule
from .BasicModule import BasicModule
import torch.nn.functional as F
import math
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."))
from config import opt
import scipy.io as sio
from torch.autograd import Variable
from torch.nn import Parameter as Parameter
from .SC import SC
import torchvision

class FBC(BasicModule):
    def __init__(self):  
        super(FBC, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])  # Remove pool5.

        self.device=torch.device("cuda")
        self.output_dim = self.JOINT_EMB_SIZE = opt.RANK_ATOMS * opt.NUM_CLUSTER #20*2048
        self.input_dim = opt.down_chennel

        self.Linear_dataproj_k = nn.Linear(opt.down_chennel, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2_k = nn.Linear(opt.down_chennel, self.JOINT_EMB_SIZE) 

        self.Linear_predict = nn.Linear(opt.NUM_CLUSTER, opt.class_num)

        self.sc = SC(beta=opt.BETA)
        if opt.res == 224:
            self.Avgpool = nn.AvgPool1d(kernel_size=196)
        elif opt.res == 448:
            self.Avgpool = nn.AvgPool1d(kernel_size=784)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data,)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        

    def forward(self, x):
        x = self.features(x) 
        bs, c, w, h = x.shape[0:4]

        bswh = bs*w*h
        x = x.permute(0,2,3,1)                  
        x = x.contiguous().view(-1,c)           

        x1 = self.Linear_dataproj_k(x)             
        x2 = self.Linear_dataproj2_k(x)

        bi = x1.mul(x2)  

        bi = bi.view(-1, 1, opt.NUM_CLUSTER, opt.RANK_ATOMS)       
        bi = torch.squeeze(torch.sum(bi, 3))                        

        bi = self.sc(bi)

        bi = bi.view(bs,h*w,-1)                                     
        bi = bi.permute(0,2,1)                                      
        bi = torch.squeeze(self.Avgpool(bi))                   

        bi = torch.sqrt(F.relu(bi)) - torch.sqrt(F.relu(-bi))      
        bi = F.normalize(bi, p=2, dim=1)

        y = self.Linear_predict(bi) 
        return y


if __name__ == '__main__':
    print(1)

