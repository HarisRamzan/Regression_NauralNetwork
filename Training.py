# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 04:24:45 2020

@author: haris
"""
import torch
import statistics
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
X_train=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TrainData.csv")
y_train=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TrainLabels.csv")
X_test=np.loadtxt("D:\\Fast University\\Semester2\\Machine Learning\\Assignments\\Assignment3\\TestData.csv") 
data=Variable(torch.FloatTensor(X_train), requires_grad=True)
labels=Variable(torch.FloatTensor(y_train), requires_grad=False)
xtest=Variable(torch.FloatTensor(X_test), requires_grad=False)
net = torch.nn.Sequential(
        torch.nn.Linear(8, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100,1),
    )
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared) 
    return rmse_val

rmsee=[]
eposhs=3000
for t in range(eposhs):
    prediction = net(data)     # input x and predict based on x
    rmsee.append(rmse(prediction.detach().numpy(),labels.detach().numpy()))
    loss = loss_func(prediction,labels)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients 
    
print("RMSE MEAN FOR ITERATIONS:",eposhs,np.mean(np.array(rmsee)))
print("RMSE Standard daviation FOR ITERATIONS:",eposhs,np.std(np.array(rmsee)))
#Freezing the weights    
for param in net.parameters():
        param.requires_grad =False
        
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared) 
    return rmse_val

#y_predicted=net(xtest)

#rms = rmse(labels,y_predicted)
#print(rms)
torch.save(net.state_dict(),"Mymodel.pt")




