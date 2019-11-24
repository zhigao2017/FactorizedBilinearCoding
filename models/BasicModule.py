#coding:utf-8
import torch
import time
import os
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."))
from config import opt

class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        #self.model_name=str(type(self))# 默认名字
        self.model_name=str(type(self)).strip('<>').split('.')[-1][:-1]
        # print(self.model_name)
        # print(os.getcwd()[:-6])

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        #print('i am loading the ' + path)
        self.load_state_dict(torch.load(path))

    def save(self, acc=None, lr=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        prefix = opt.model_save_path + self.model_name + '_'
        #print(prefix)
        #print(os.getcwd())
        name = prefix + time.strftime('%m%d_%H%M_')  + '_acc_' \
            + str(acc).replace('.','_')  + 'lr' + lr + '.pth'
        #print(name)
        torch.save(self.state_dict(), name)
        return name



if __name__ == '__main__':
    a = BasicModule()

    name = a.save()
    print(name)