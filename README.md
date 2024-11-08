# LaserMon

Software templates for edge-AI sensor fusion and monitoring of laser-based additive manufacturing

## Introduction

In this repository, you will find code templates for very high speed sensor fusion for the monitoring
of laser-based additive manufacturing. These templates show how data from a relatively slow,
high-reolution sensor can be fused with data from fast, low-resolution sensor. An adaptive weight
model is used to perform this sensor fusion, and the model size is kept small so that it can run at
100 kHz.

Note that this code is not designed to be run directly. It is provided as a template for how this
edge-AI sensor fusion technique could be applied. The idea is for the code to be studied and adapted
to fit your use case.

Within this repository, you will find code for two use cases: metal laser powder bed fusion, and
polymer selective laser sintering. Files for the first use case are marked with the SLM acronym in
their file name, while the second use case has the acronym SLS in their file names. The files are
divided into three directories:

1. **preprocessing**: This directory contains various scripts to help synchronize the sensor data
   collected in our use cases, and to calibrate the cameras used in our use cases to the coordinate
   system of the 3D printer's laser. These scripts are likely to be of limited value to you unless
   you are working with similar datasets as the ones used in our use case.

2. **training**: This directory contains the code for the adaptive weight models (LaserMon_model.py
   and AWN_layers.py), for creating the training sets for both use cases, and a python script to
   output the weights and biases of the adaptive weight network to a C file for hih-speed inference.

3. **inference**: This directory contains the C code and Makefile for compiling and running the
   adaptive weight network at high speeds. Unlike the other directories, the C code provided here
   is well commented and described.

## Contact

If you run into any questions regarding the code, please contact me at <brian.booth@ugent.be>