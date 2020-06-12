import os
import sys
import torch
import torch.nn as nn
import numpy as np

class PhaseFunction(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        """such dimensions are in PFNN model"""
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = output_dim

        self.phase_function_output_dim = self.input_dim*self.hidden_dim + self.hidden_dim \
                                         + self.hidden_dim*self.hidden_dim + self.hidden_dim \
                                         + self.hidden_dim*self.output_dim + self.output_dim

        self.layer1 = nn.Linear(4, self.phase_function_output_dim, bias=False)

    def forward(self, phase):
        w = phase#[0,1]
        one = torch.ones_like(w,requires_grad=True)
        input = torch.cat([one, w, w**2, w**3], 2)
        return self.layer1(input)

class PFNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        """re-implementation of PFNN"""
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = output_dim
        self.drop_prob = args.drop_prob
        """Nerual Network block"""
        self.phase_function = PhaseFunction(self.input_dim, self.output_dim, args)


    def forward(self, input, phase):
        weight_bias = self.phase_function(phase)

        self.weight1 = torch.reshape(weight_bias[:,:,:self.input_dim*self.hidden_dim], (-1, self.input_dim, self.hidden_dim))
        self.bias1 = weight_bias[:,:,self.input_dim*self.hidden_dim : self.input_dim*self.hidden_dim + self.hidden_dim]
        last = self.input_dim*self.hidden_dim + self.hidden_dim

        self.weight2 = torch.reshape(weight_bias[:, :, last : last + self.hidden_dim*self.hidden_dim], (-1, self.hidden_dim, self.hidden_dim))
        self.bias2 = weight_bias[:, :, last + self.hidden_dim*self.hidden_dim:last + self.hidden_dim*self.hidden_dim+ self.hidden_dim]
        last = last + self.hidden_dim*self.hidden_dim+ self.hidden_dim

        self.weight3 = torch.reshape(weight_bias[:, :, last:last + self.hidden_dim*self.output_dim], (-1,self.hidden_dim, self.output_dim))
        self.bias3 = weight_bias[:,:,last + self.hidden_dim*self.output_dim:]

        """
        print()
        print("Size Check")
        print(self.weight1.size(),self.bias1.size())
        print(self.weight2.size(),self.bias2.size())
        print(self.weight3.size(),self.bias3.size())
        print()
        """

        "layer1"
        x = nn.Dropout(p=self.drop_prob)(input)
        x = torch.matmul(x,self.weight1) + self.bias1
        x = nn.ELU()(x)
        "layer2"
        x = nn.Dropout(p=self.drop_prob)(x)
        x = torch.matmul(x, self.weight2) + self.bias2
        x = nn.ELU()(x)
        "layer3"
        x = nn.Dropout(p=self.drop_prob)(x)
        x = torch.matmul(x, self.weight3) + self.bias3

        return x

    def save_model(self, iter, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        ckpt_path_phasefunc = dir + str(iter) + "pfnn.pth.tar"
        torch.save(self.phase_function.state_dict(), ckpt_path_phasefunc)

    def load_model(self, filename):
        self.phase_function.load_state_dict(torch.load(filename))


"""
#test dimension
from config import CONFIGS
input_dim = 5
output_dim = 8
batch_size = 100
args = CONFIGS["pfnn"]
#pf = PhaseFunction(input_dim, output_dim, args)
model = PFNNModel(input_dim, output_dim, args)

#print(pf(dumy_phase).size())
#run time check
import time
for i in range(10):
    s = time.time()
    dumy_input = torch.randn(batch_size, 1, input_dim)
    dumy_phase = torch.randn(batch_size, 1, 1)
    #print(model(dumy_input, dumy_phase).size())
    print(1/(time.time()-s),"hz")
"""
