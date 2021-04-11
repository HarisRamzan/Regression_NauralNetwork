# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:58:18 2020

@author: haris
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
#from torchviz import make_dot
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hc1 = torch.nn.Linear(8, 14)  
        self.hc2 = torch.nn.Linear(14,12)
        self.hc3 = torch.nn.Linear(12,8)
        self.hc4 = torch.nn.Linear(8, 4)
        self.oupt=torch.nn.Linear(4, 1)
#        torch.nn.init.xavier_uniform_(self.hc1.weight)  # glorot
#        torch.nn.init.zeros_(self.hc1.bias)
#        torch.nn.init.xavier_uniform_(self.hc2.weight)
#        torch.nn.init.zeros_(self.hc2.bias)
#        torch.nn.init.xavier_uniform_(self.hc3.weight)
#        torch.nn.init.zeros_(self.hc3.bias)
#        torch.nn.init.xavier_uniform_(self.hc4.weight)
#        torch.nn.init.zeros_(self.hc4.bias)
#        torch.nn.init.xavier_uniform_(self.oupt.weight)
#        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hc1(x))
        z = torch.tanh(self.hc2(z))
        z = torch.tanh(self.hc3(z))
        z = torch.tanh(self.hc4(z))
        z = self.oupt(z)  # no activation, aka Identity()
        return z

def accuracy(model, data_x, data_y, pct_close):
  n_items = len(data_y)
  X = torch.Tensor(data_x)  # 2-d Tensor
  Y = torch.Tensor(data_y)  # actual as 1-d Tensor
  oupt = model(X)       # all predicted as 2-d Tensor
  pred = oupt.view(n_items)  # all predicted as 1-d
  n_correct = torch.sum((torch.abs(pred - Y) < torch.abs(pct_close * Y)))
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  return result 
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared) 
    return rmse_val
def main():
    #reading csv files
    X_train=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TrainData.csv")
    y_train=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TrainLabels.csv")
    X_test=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TestData.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.20)
    #x_train_tensor = torch.from_numpy(X_train).float().to(device)
    #y_train_tensor = torch.from_numpy(y_train).float().to(device)
    torch.manual_seed(1);  np.random.seed(1)
#    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#    print(a, b)
    net = Net()
    net = net.train()
    bat_size = 5
    loss_func = torch.nn.MSELoss()  # Mean squared error
    optimizer =  torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    n_items = len(X_train)
    batches_per_epoch = n_items // bat_size
    max_batches = 1000 * batches_per_epoch
    for b in range(max_batches):
        curr_bat = np.random.choice(n_items, bat_size,replace=False)
        X = torch.Tensor(X_train[curr_bat])
        Y = torch.Tensor(y_train[curr_bat]).view(bat_size,1)
        optimizer.zero_grad()
        oupt = net(X)
        #print(rmse(y_train,oupt))
        loss_obj = loss_func(oupt,Y)
        loss_obj.backward()  # Compute gradients
        optimizer.step()     # Update weights and biases
        
        if b % (max_batches // 10) == 0:
            print("batch = %6d" % b, end="")
            print("  batch loss = %7.4f" % loss_obj.item(), end="")
            net = net.eval()
            acc = accuracy(net, X_train,y_train,0.15)
            net = net.train()
            print("  accuracy = %0.2f%%" % acc)
if __name__ == '__main__':
    main()

