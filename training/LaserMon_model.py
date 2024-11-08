#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWN with single layer

Created on Fri Oct  4 15:32:07 2024

@author: bbooth
"""

import h5py
import time
import torch
import itertools
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from AWN_layers import DotMaskLayer

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class LaserMon_Dataset(Dataset):
    
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
    
    def get_outputs_stats(self):
        
        MU_out = torch.mean(self.outputs, dim=0)
        STD_out = torch.std(self.outputs, dim=0)
        return MU_out, STD_out
    
    def normalize(self):
        
        MU_in = torch.atleast_2d(torch.mean(self.inputs, dim=0))
        MU_in = MU_in.repeat(self.inputs.shape[0],1)
        STD_in = torch.atleast_2d(torch.std(self.inputs, dim=0))
        STD_in = STD_in.repeat(self.inputs.shape[0],1)
        self.inputs = (self.inputs - MU_in) / STD_in
        
        MU_aux = torch.atleast_2d(torch.mean(self.aux_inputs, dim=0))
        MU_aux = MU_aux.repeat(self.aux_inputs.shape[0],1)
        STD_aux = torch.atleast_2d(torch.std(self.aux_inputs, dim=0))
        STD_aux = STD_aux.repeat(self.aux_inputs.shape[0],1)
        self.aux_inputs = (self.aux_inputs - MU_aux) / STD_aux
        
    def shuffle(self):
        
        idx = np.random.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[idx,:]
        self.aux_inputs = self.aux_inputs[idx,:]
        self.outputs = self.outputs[idx,:]


class LaserMon_Model(nn.Module):
    
    def __init__(self, n_aux, n_in, n_hidden, n_out, MU_out, STD_out):
        
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.MU = torch.atleast_2d(MU_out).cuda()
        self.STD = torch.atleast_2d(STD_out).cuda()
        
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
        
        #self.loss = nn.MSELoss()
        self.trainer = optim.Adam(itertools.chain(*params), lr=l_rate)
        self.scheduler = ExponentialLR(self.trainer, gamma=0.9)
        
        
    def forward(self, inputs, aux_in):
        
        W = self.aux_netowrk(aux_in)
        H = self.awn_network(inputs, W, aux_in[:,-1])
        return self.pred_network(H)
        
    
    def loss(self, outputs, targets):
        
        MU_out = self.MU.repeat(targets.shape[0],1)
        STD_out = self.STD.repeat(targets.shape[0],1)
        targets = (targets - MU_out) / STD_out
        outputs = (outputs - MU_out) / STD_out
        
        loss = torch.mean((outputs - targets)**2)
        return loss
    
    
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
                aloss, current = np.sqrt(iter_loss.item()), batch * dataloader.batch_size + len(X)
                print(f"loss: {aloss:>7f}  [{current:>6d}/{size:>6d}]")
        
        self.scheduler.step()
    
    
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
                test_loss += self.loss(pred, y.cuda()).item()
                correct += (torch.abs(pred[:,0] - y.cuda()[:,0]) < 0.18912).type(torch.float).sum().item()
                
        test_loss /= num_batches
        test_loss = np.sqrt(test_loss)
        correct /= size  # A result is correct if its error is less than 10%
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        
    def get_predictions(self, inputs, aux_in):
        self.eval()
        inputs = inputs.cuda()
        aux_in = aux_in.cuda()
        with torch.no_grad():
            tic = time.time_ns() 
            preds = self.forward(inputs, aux_in)
            toc = time.time_ns()
        
        t = (toc-tic) / 1000
        print(f"Time: {(t):>0.1f} microseconds, Number of predictions: {inputs.shape[0]:>6d}")
        print(f"Time per prediction: {((t) / inputs.shape[0]):>0.3f} microseconds")
        return preds
        
        
def main():
    pass

if __name__ == '__main__':
    main()
    