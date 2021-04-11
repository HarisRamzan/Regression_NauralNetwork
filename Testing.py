# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:59:14 2020

@author: haris
"""
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared) 
    return rmse_val

net = torch.nn.Sequential(
        torch.nn.Linear(8, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100,1),
    )

X_test=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TestData.csv") 
testdata=Variable(torch.FloatTensor(X_test), requires_grad=True)
net.load_state_dict(torch.load("Mymodel.pt"))
#print(net(testdata))
np.savetxt("i192118_Predictions.csv",net(testdata).detach().numpy())

#uncomment below lines for calculating RMSE
#rms = rmse(labels,net(testdata).detach().numpy())
#print(rms)