
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"

#ifndef _HDF5_READ_H_
#define _HDF5_READ_H_

/* HDF5 reading functions */
float* my_malloc_float(size_t N);
float* load_inputs(char* file_loc, int* n_samples, int n_inputs);
float* load_aux_inputs(char* file_loc, int* n_samples, int n_aux_inputs);
void get_sample(float* input_buffer, float* sample_vec, int n_inputs, int sample_id);

#endif
