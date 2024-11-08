#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Design the SLS prediction model

Created on Tue Sep 24 10:26:33 2024

@author: bbooth
"""

import h5py
import torch
import itertools
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch.nn import functional as F
from AWN_layers import DotMaskLayer

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class SLS_Dataset(Dataset):
    
    def __init__(self, data_loc, transform=None, target_transform=None):
        
        fp = h5py.File(data_loc, 'r')
        IN = np.array(list(fp['inputs']))
        AUX = np.array(list(fp['aux_inputs']))
        OUT = np.array(list(fp['outputs']))
        fp.close()
        
        self.inputs = torch.from_numpy(IN.astype('float32'))
        self.aux_inputs = torch.from_numpy(AUX.astype('float32'))
        self.outputs = torch.from_numpy(OUT.astype('float32'))
    
        
    def __len__(self):
        return self.inputs.shape[0]
        
    
    def __getitem__(self, idx):
        return self.inputs[idx,:], self.aux_inputs[idx,:], self.outputs[idx,:]
    
    
    def normalize(self):
        
        MU_in = torch.mean(self.inputs, dim=0)
        STD_in = torch.std(self.inputs, dim=0)
        self.inputs = (self.inputs - MU_in) / STD_in
        
        MU_aux = torch.mean(self.aux_inputs, dim=0)
        STD_aux = torch.std(self.aux_inputs, dim=0)
        self.aux_inputs = (self.aux_inputs - MU_aux) / STD_aux


class SLS_Prediction_Model(nn.Module):
    
    def __init__(self, n_aux, n_in, n_hidden, n_out):
        
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Optimization parameters
        l_rate = 0.1
        
        self.aux_netowrk = nn.Linear(n_aux, n_hidden*(n_in+1))
        self.awn_network = DotMaskLayer()
        self.pred_network = nn.Linear(n_hidden, n_out)
        
        self.aux_netowrk.cuda()
        self.awn_network.cuda()
        self.pred_network.cuda()
        params = [self.aux_netowrk.parameters(), self.awn_network.parameters(),
                  self.pred_network.parameters()]
        
        self.loss = nn.MSELoss()
        self.trainer = optim.Adam(itertools.chain(*params), lr=l_rate)
        
        
    def forward(self, inputs, aux_in):
        
        #A = self.flatten(aux_in)
        W = self.aux_netowrk(aux_in)
        # print(aux_in.size())
        # print(aux_in[:,-1].size())
        # print(inputs.size())
        # print(W.size())
        H = self.awn_network(inputs, W, aux_in[:,-1])
        return self.pred_network(H)
        
    
    def train_loop(self, dataloader):
        
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.train()
        
        for batch, (X, A, y) in enumerate(dataloader):
            
            # Compute prediction and loss
            pred = self.forward(X.cuda(), A.cuda())
            iter_loss = self.loss(pred, y.cuda())

            # Backpropagation
            iter_loss.backward()
            self.trainer.step()
            self.trainer.zero_grad()

            if batch % 100 == 0:
                aloss, current = iter_loss.item(), batch * dataloader.batch_size + len(X)
                print(f"loss: {aloss:>7f}  [{current:>5d}/{size:>5d}]")
    
    
    def test_loop(self, dataloader):
    
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        #print(num_batches)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, A, y in dataloader:
                pred = self.forward(X.cuda(), A.cuda())
                test_loss = self.loss(pred, y.cuda()).item()
                #test_loss += (np.sqrt(tmp_loss) / 64)
                #print(np.sqrt(tmp_loss) / 64) 
                #print(y)
                correct += (torch.abs(pred[:,0] - y.cuda()[:,0]) < 1).type(torch.float).sum().item()
                
        test_loss /= num_batches
        #test_loss = np.sqrt(test_loss)
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

def main():
    
    train_loc = '/scratch/bbooth/P3AI/training_set.h5'
    
    training_set = SLS_Dataset(train_loc)
    training_set.normalize()
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    
    model = SLS_Prediction_Model(3, 100, 50, 2)
    n_epocs = 100
    
    for ii in range(n_epocs):
        print(f"Epoch {ii+1}\n-------------------------------")
        model.train_loop(training_loader)
        model.test_loop(training_loader)
    print("Done!")
    
    torch.save(model.state_dict(), '/scratch/bbooth/P3AI/model_weights.pth')
    

if __name__ == '__main__':
    main()