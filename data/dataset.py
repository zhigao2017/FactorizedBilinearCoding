#coding:utf-8
import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from config import opt

def default_loader(path):
    #print(path)
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words) == 2:
                imgs.append((words[0], int(words[1])))
            elif len(words) == 3:
                tmp = words[0] + ' ' + words[1]
                imgs.append((tmp, int(words[-1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data_pth = opt.data_path
        #img = self.loader('/data/guijun/caffe-20160312/examples/compact_bilinear/cub/images/' + fn)
        img = self.loader(data_pth + fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)




        
# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# from PIL import Image
# from torch.utils import data
# import numpy as np
# import torch
# #from torchvision import transforms as T
# import sys, os
# from config import opt 
# import matplotlib.pyplot as plt
# import six.moves.cPickle as pickle
# import random
# import warnings
# warnings.filterwarnings('ignore')

# class RGBD(data.Dataset):

#     def __init__(self, root, train=False, val=False, test=False, split=opt.split):
#         '''
#         get the data and split them into train, val and test subset;
#         '''
#         print('split:', split)
#         self.train = train
#         self.val = val
#         self.test = test

#         imgs = []
#         #stage = '1' if train else '2' if val else '3' if test else '4'
#         #val -> train ,discard the val
#         stage = '1' if train else '2' if val else '3' if test else '4'
#         print('stage: ',('train' if stage=='1' else 'val' 
#             if stage=='2' else 'test' if stage=='3' else 'UNKNOWNSTAGE'))
#         #print('csvroot:',root)
#         f = open(root, 'r')

#         f.readline()
#         lines = f.readlines()
#         for line in lines:
#             contents = line.strip('\n\r').split(',')
#             #print('contents',contents)
#             #img_name   cat_label   ins_label   ins_set split 0-9

#             if stage == '1':#train
#                 if contents[4+split] == '1':
#                     if int(contents[0].split('_')[-2])%opt.train_n_th != 1:
#                         continue
#                     #print(int(contents[0].split('_')[-2]))
#                     item = {'rgb_path' : contents[0], 
#                             'depth_path' : contents[0][:-8] + 'depthcrop.png',
#                             'mask_path' : contents[0][:-8] + 'maskcrop.png',
#                             'cat_label' : int(contents[1]),
#                             }
#                     imgs.append(item)   
#             elif stage == '3':#test
#                 if contents[4+split] == '3' :
#                     if int(contents[0].split('_')[-2])%opt.test_n_th != 1:
#                         continue
#                     item = {'rgb_path' : contents[0], 
#                             'depth_path' : contents[0][:-8] + 'depthcrop.png',
#                             'mask_path' : contents[0][:-8] + 'maskcrop.png',
#                             'cat_label' : int(contents[1]),
#                             }
#                     imgs.append(item)          
        


#         self.imgs = imgs
        

#     def __getitem__(self, index):
#         '''
#         return a picture according to the given index once time;
#         '''
#         rgb_path = '/'.join(opt.csv_rgb256_path.split('/')[:-1]) + '/' + self.imgs[index]['rgb_path']
#         rgb_data = Image.open(rgb_path)
#         if self.train:
#             rgb_flip_random = random.random()
#             if rgb_flip_random > 0.5:
#                 rgb_data = rgb_data.transpose(Image.FLIP_LEFT_RIGHT)
#         with open(opt.rgb_mean_path, 'rb') as f:
#             rgb_mean = pickle.load(f)
#             rgb_mean = torch.from_numpy(rgb_mean)
#             rgb_data = np.asarray(rgb_data)
#             rgb_data = np.transpose(rgb_data ,(2, 0, 1))
#             rgb_data = torch.from_numpy(rgb_data).double()
#             rgb_data = (rgb_data-rgb_mean)

#             rgb_data = rgb_data/255.0
#             left = random.randint(0, opt.resolutionPlus-opt.resolution)
#             top = random.randint(0, opt.resolutionPlus-opt.resolution)
#             rgb_data = rgb_data[:,left:left+opt.resolution,top:top+opt.resolution].float()


#         depth_path = '/'.join(opt.csv_depth256_path.split('/')[:-1]) + '/' + self.imgs[index]['depth_path']
#         depth_data = Image.open(depth_path)
#         if self.train:
#             depth_flip_random = random.random()
#             if depth_flip_random > 0.5:
#                 depth_data = depth_data.transpose(Image.FLIP_LEFT_RIGHT)
#         with open(opt.depth_mean_path, 'rb') as f:
#             depth_mean = pickle.load(f)
#             depth_mean = torch.from_numpy(depth_mean)          
#             depth_data = np.asarray(depth_data)
#             depth_data = np.transpose(depth_data ,(2, 0, 1))
#             depth_data = torch.from_numpy(depth_data).double()
#             depth_data = (depth_data - depth_mean)
#             depth_data = depth_data/255.0
#             left = random.randint(0, opt.resolutionPlus-opt.resolution)
#             top = random.randint(0, opt.resolutionPlus-opt.resolution)
            
#             #depth_data = torch.FloatTensor(depth_data[:,left:left+opt.resolution,top:top+opt.resolution])
#             depth_data = depth_data[:,left:left+opt.resolution,top:top+opt.resolution].float()

#             #print(depth_data)
#         label = self.imgs[index]['cat_label']
#         return (rgb_data, depth_data), label
        


#     def __len__(self):
#         return len(self.imgs)

# if __name__ == '__main__':
#     root = opt.csv_path
#     # for i in range(10): 
#     #     rgbd = RGBD(root,train=True,split=i)
#     #     print('train len:',len(rgbd.imgs))
#     #     rgbd = RGBD(root,test=True,split=i)
#     #     print('test len:',len(rgbd.imgs))
#     rgbd = RGBD(root,train=True,split=0)
#     print('train len:',len(rgbd.imgs))  
#     a = rgbd.imgs[:10] 
#     # for i in a:
#     #     print(i)
#     #print(label)
#     #print(len(rgbd))