#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a neural network trained by pyTorch to weight and bias matrices
in C format

Created on Wed Aug 30 14:26:38 2023

@author: bbooth
"""

import h5py
import torch
import LaserMon_model as LMM
import numpy as np

def load_image_features(data_loc):
    
    fp = h5py.File(data_loc, 'r')
    AUX = np.array(list(fp['aux_inputs']))
    IN = np.array(list(fp['inputs']))
    fp.close()
    
    return IN, AUX


def print_out_row(fp, weights, bias):
    fp.write('{')
    for ii in range(len(weights)):
        if ii > 0 and ii % 4 == 0:
            fp.write(' \n\t')
        fp.write(str(weights[ii].item()) + ', ')
    fp.write(str(bias.item()) + '}')


def main():
    model_loc = '/scratch/bbooth/FAIR/model_weights_prop.pth'
    out_loc = '/scratch/bbooth/FAIR/model_params.c'

    model = torch.load(model_loc, weights_only=False)
    model.eval()
    
    aux_weights = model.aux_netowrk.weight.cpu()
    aux_bias = model.aux_netowrk.bias.cpu()
    
    pred_weights = model.pred_network.weight.cpu()
    pred_bias = model.pred_network.bias.cpu()
    
    fp = open(out_loc, 'w')
    
    #fp.write('\n#include <math.h> /* for the tanh function */ \n\n')
    
    #fp.write('#ifndef _MODEL_H_ \n')
    #fp.write('#define _MODEL_H_ \n')
    
    #fp.write('#DEFINE INPUT_SZ ' + str(model[0].weight.shape[1]) + ' /* Number of input features */ \n')
    #fp.write('#DEFINE HIDDEN_SZ ' + str(model[0].weight.shape[0]) + ' /* Number of hidden nodes */ \n')
    #fp.write('#DEFINE OUTPUT_SZ ' + str(model[2].weight.shape[0]) + ' /* Number of predicted outputs */ \n')
    #fp.write('\n')
    
    fp.write('/* Weights and biases for the processing the contextual features */ \n')    
    fp.write('const float layerAUX [' + str(aux_weights.shape[0]) + '][' + str(aux_weights.shape[1]+1) + '] = {')
    for ii in range(aux_weights.shape[0]):
        if ii > 0:
            fp.write(', \n\t')
        print_out_row(fp, aux_weights[ii,:], aux_bias[ii])
    fp.write('};\n\n')
    
    
    fp.write('/* Weights and biases of the fully connected output layer. */ \n')
    fp.write('const float layerOUT [' + str(pred_weights.shape[0]) + '][' + str(pred_weights.shape[1]+1) + '] = {')
    for ii in range(pred_weights.shape[0]):
        if ii > 0:
            fp.write(', \n\t')
        print_out_row(fp, pred_weights[ii,:], pred_bias[ii])
    fp.write('};\n\n')
    
    # Load training set & compute normalization values
    data_loc = '/scratch/bbooth/FAIR/datasets/training_data.h5'
    inputs, aux_in = load_image_features(data_loc)
    
    mu_inputs = np.mean(inputs, axis=0)
    std_inputs = np.std(inputs, axis=0)
    
    mu_aux = np.mean(aux_in, axis=0)
    std_aux = np.std(aux_in, axis=0)
    
    fp.write('/* Normalization constants for the inputs */')
    fp.write('const float normINP [2][' + str(len(mu_inputs)) + '] = {')
    print_out_row(fp, mu_inputs[:-1], mu_inputs[-1])
    fp.write(', \n\t')
    print_out_row(fp, std_inputs[:-1], std_inputs[-1])
    fp.write('};\n\n')
    
    fp.write('/* Normalization constants for the auxiliary inputs */')
    fp.write('const float normAUX [2][' + str(len(mu_aux)) + '] = {')
    print_out_row(fp, mu_aux[:-1], mu_aux[-1])
    fp.write(', \n\t')
    print_out_row(fp, std_aux[:-1], std_aux[-1])
    fp.write('};\n\n')
    
    
    #fp.write('#endif \n')
    
    fp.close()
    
    # print(model)
    # print(model[0].weight.shape)
    # print(model[0].bias.shape)
    # print(model[2].weight.shape)
    # print(model[2].bias.shape)
    
if __name__ == '__main__':
    main()