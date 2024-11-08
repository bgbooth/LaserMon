#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make SLM Prediction Model

Created on Fri Sep 27 14:33:50 2024

@author: bbooth
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import LaserMon_model as LMM
import DiodeModel as DM
from torch.utils.data import DataLoader

class Test3_Dataset(Dataset):
    
    def __init__(self, data_loc, transform=None, target_transform=None):
        
        fp = h5py.File(data_loc, 'r')
        IN = np.array(list(fp['inputs']))
        AUX = np.array(list(fp['aux_inputs']))
        OUT = np.array(list(fp['outputs']))
        fp.close()
        AUX = AUX[:,3:]
        
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

def main():
    
    train_loc = '/scratch/bbooth/FAIR/datasets/training_data.h5'
    
    training_set = LMM.LaserMon_Dataset(train_loc)
    training_set.normalize()
    training_set.shuffle()
    MU, STD = training_set.get_outputs_stats()
    training_loader = DataLoader(training_set, batch_size=128, shuffle=True)
    
    model = LMM.LaserMon_Model(6, 40, 20, 3, MU, STD)
    n_epocs = 100
    print(STD)
    
    for ii in range(n_epocs):
        print(f"Epoch {ii+1}\n-------------------------------")
        model.train_loop(training_loader)
        model.test_loop(training_loader)
    print("Done!")
    
    torch.save(model, '/scratch/bbooth/FAIR/model_weights_prop.pth')
    
    comp_set3 = Test3_Dataset(train_loc)
    comp_set3.normalize()
    comp_set3.shuffle()
    MU3, STD3 = comp_set3.get_outputs_stats()
    comp3_loader = DataLoader(comp_set3, batch_size=128, shuffle=True)

    model = LMM.LaserMon_Model(3, 40, 20, 3, MU3, STD3)
    n_epocs = 100
    print(STD3)
    
    for ii in range(n_epocs):
        print(f"Epoch {ii+1}\n-------------------------------")
        model.train_loop(comp3_loader)
        model.test_loop(comp3_loader)
    print("Done!")
    
    torch.save(model, '/scratch/bbooth/FAIR/model_weights_comp3.pth')
    
    comp_set2 = DM.Diode_Dataset(train_loc)
    comp_set2.normalize()
    comp_set2.shuffle()
    MU2, STD2 = comp_set2.get_outputs_stats()
    comp2_loader = DataLoader(comp_set2, batch_size=128, shuffle=True)

    model = DM.Diode_Model(40, 20, 3, MU2, STD2)
    n_epocs = 100
    print(STD2)
    
    for ii in range(n_epocs):
        print(f"Epoch {ii+1}\n-------------------------------")
        model.train_loop(comp2_loader)
        model.test_loop(comp2_loader)
    print("Done!")
    
    torch.save(model, '/scratch/bbooth/FAIR/model_weights_comp2.pth')

if __name__ == '__main__':
    main()