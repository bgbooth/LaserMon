#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWN dot product with mask layer

Created on Tue Sep 24 10:05:36 2024

@author: bbooth
"""

import torch
from torch import nn
from torch.nn import functional as F
#from d2l import torch as d2l

class DotLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, AUX):
        
        # Make output tensor
        H = int(AUX.size(dim=1) / (1+X.size(dim=1)))
        dot_prod = torch.zeros([X.size(dim=0), H]).cuda()
                
        for ii in range(X.size(dim=0)):
            
            # Make weights and bias matrix
            W = torch.reshape(AUX[ii,:], (1+X.size(dim=1), H))
        
            # Add one to input vector (for bias calculation)
            Y = torch.cat((X[ii,:], torch.tensor([1]).cuda()))
        
            # Apply layer computation
            dot_prod[ii,:] = torch.matmul(Y, W) 
        
        # Apply trasition function
        return F.tanh(dot_prod)


class DotMaskLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, AUX, K):
        
        # Make output tensor
        H = int(AUX.size(dim=1) / (1+X.size(dim=1)))
        dot_prod = torch.zeros([X.size(dim=0), H]).cuda()
        
        M = int(X.size(dim=1) / 2)
        #print(M)
        
        for ii in range(X.size(dim=0)):
            
            # Make weights and bias matrix
            W = torch.reshape(AUX[ii,:], (1+X.size(dim=1), H))
        
            # Zero out the weights related to unnecessary inputs
            idx = int(K[ii].item())
            W[idx:M,:] = 0.0
            W[M+idx:,:] = 0.0
        
            # Add one to input vector (for bias calculation)
            Y = torch.cat((X[ii,:], torch.tensor([1]).cuda()))
        
            # Apply layer computation
            dot_prod[ii,:] = torch.matmul(Y, W) 
        
        # Apply trasition function
        return F.tanh(dot_prod)
    

def main():
    pass

if __name__ == '__main__':
    main()