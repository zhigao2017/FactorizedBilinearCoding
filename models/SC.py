import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from config import opt

class SC(nn.Module):
    def __init__(self,beta):
        super(SC, self).__init__()
        self.device=torch.device("cuda")
        self.beta=beta
        # if opt.learn_beta:
        #     self.beta = nn.Parameter(torch.tensor(beta))
        # else:
        #     self.beta = beta
        #self.B=nn.Parameter(torch.randn(10,20))#c*c - > num_cluster

  


    def forward(self, input):
       
        zero = torch.zeros(input.shape).to(self.device)
        output = torch.mul(torch.sign(input),torch.max((torch.abs(input)-self.beta/2),zero))
        

        return output

if __name__ == '__main__':
    a = SC(1)

    input = torch.randn(2,3)
    print(input)
    out = a(input)
    print(out)
