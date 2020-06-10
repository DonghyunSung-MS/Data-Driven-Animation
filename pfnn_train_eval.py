import random
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from model.PFNN import PFNNModel
from model.config import CONFIGS
from utils.customDataLoader import PFNNDataSet

args = CONFIGS["pfnn"]

"""Set random seed"""
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""Loading Data"""
dataset = PFNNDataSet('./data/pfnn/database_from_multi_small.npz', args)
validation_split = .2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if args.shuffle :
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

#print(train_loader)
"""Creating Model"""
input_dim = 342
output_dim = 311

model = PFNNModel(input_dim, output_dim, args)
mse = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

from torchsummary import summary
device = torch.device("cuda" if args.gpu else "cpu")
model.to(device)#this is for torchsummary
summary(model,[(1,input_dim),(1,1)])

"""train start"""
if args.wandb:
    wandb.init(project="data_driven_animation")

model.train()
import time
for epoch in range(args.epoch):
    s = time.time()
    for i, data in enumerate(train_loader):
        inputs, phases, outputs = data
        if args.gpu:
            inputs, phases, outputs = Variable(inputs).to(device), Variable(phases).to(device), Variable(outputs).to(device)
        else:
            inputs, phases, outputs = Variable(inputs), Variable(phases), Variable(outputs)

        pred_outputs = model(inputs, phases)
        l1_norm = torch.norm(model.phase_function.layer1.weight,p=1)
        loss = mse(outputs, pred_outputs) + args.lasso_coef*l1_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(time.time() - s)
        break
        if i%log_interval==0: #assume log and save model interval is same
            model.save_model(i,args.save_dir)
            if args.wandb:
                wandb.log({"train_iter":i,
                           "train_loss":loss,
                           "train_l1_norm":l1_norm})

"""
model.eval()
for epoch in range(args.test_epoch):
    s = time.time()
    for i, data in enumerate(valid_loader):
        inputs, phases, outputs = data
        if args.gpu:
            inputs, phases, outputs = Variable(inputs).to(device), Variable(phases).to(device), Variable(outputs).to(device)
        else:
            inputs, phases, outputs = Variable(inputs), Variable(phases), Variable(outputs)

        pred_outputs = model(inputs, phases)
        l1_norm = torch.norm(model.phase_function.layer1.weight,p=1)
        loss = mse(outputs, pred_outputs) + args.lasso_coef*l1_norm
        print(time.time() - s)
        if i%log_interval==0: #assume log and save model interval is same
            model.save_model(i,args.save_dir)
            if args.wandb:
                wandb.log({"test_iter":i,
                           "test_loss":loss})
"""
