
#include "hdf5_read.h"

/* HDF5 group names */
#define INPUTS_KEY "inputs"
#define AUX_INPUTS_KEY "aux_inputs"

/* Maximum input feature numbers for different models */
#define MAX_STR_SZ 256

/* Wrapper function for H5Fopen */
hid_t my_h5f_open(const char* filename, unsigned flags, hid_t fapl_id) {

  hid_t ret_val = -1;

  ret_val = H5Fopen(filename, flags, fapl_id);
  if (ret_val == H5I_INVALID_HID) {
    perror("H5Fopen failed");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/* Wrapper function for H5Dopen2 */
hid_t my_h5d_open2(hid_t loc_id, const char* name, hid_t dapl_id) {

  hid_t ret_val = -1;

  ret_val = H5Dopen2(loc_id, name, dapl_id);
  if (ret_val == H5I_INVALID_HID) {
    perror("H5Dopen2 failed");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/* Wrapper function for H5Sget_simple_extent_npoints */
hsize_t get_matrix_size(hid_t mtx_ptr) {

  int ret_val = -1;
  hsize_t mtx_sz[2] = {0,0}; /* The matrix size buffer */
  hid_t dspace = -1;         /* Data space for the matrix */

  dspace = H5Dget_space(mtx_ptr);
  if (dspace == H5I_INVALID_HID) {
    perror("Failure to get matrix's data space");
    exit(EXIT_FAILURE);
  }

  ret_val = H5Sget_simple_extent_dims (dspace, mtx_sz, NULL);	
  if (ret_val != 2) {
    perror("Failure to read HDF5 matrix size");
    exit(EXIT_FAILURE);
  }

  return mtx_sz[0] * mtx_sz[1];
  
}

/* Wrapper function for malloc */
float* my_malloc_float(size_t N) {

  float* ptr = NULL;

  ptr = (float*) malloc(N);
  if (ptr == NULL) {
    perror("malloc failure");
    exit(EXIT_FAILURE);
  }

  return ptr;
  
}

/* Wrapper function for H5Dread */
herr_t my_h5d_read(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
		   hid_t dxpl_id, void * buf) {

  herr_t ret_val = -1;

  ret_val = H5Dread(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf);
  if (ret_val < 0) {
    perror("H5D read failed");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/* Wrapper function for H5Dclose */
herr_t my_h5d_close(hid_t data_id) {

  herr_t ret_val = -1;

  ret_val = H5Dclose(data_id);
  if (ret_val < 0) {
    perror("H5Dclose failed");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/* Wrapper function for H5Fclose */
herr_t my_h5f_close(hid_t file_id) {

  herr_t ret_val = -1;

  ret_val = H5Fclose(file_id);
  if (ret_val < 0) {
    perror("H5Fclose failed");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/* Wrapper function for sprintf */
int my_path_sprintf(char* out_buf, char* format, int layer_num) {

  int ret_val = -1;

  ret_val = sprintf(out_buf, format, layer_num);
  if (ret_val < 0) {
    perror("sprintf failed - path");
    exit(EXIT_FAILURE);
  }

  return ret_val;
  
}

/**
 * Local function to read a matrix with the given name from an HDF5 file
 * into a dynamically-allocated buffer while also prviding its size.
 *
 * Arguments:
 *   - path: string with the path to the HDF5 file
 *   - mtx_name: string with the name of the matrix to load
 *   - N_in: pointer to a buffer in which to save the size of the returned buffer
 *
 * Returns:
 *   - The buffer containing the values of the loaded matrix
 */
float* load_matrix(char* path, const char* mtx_name, unsigned long* N_in) {

  hid_t file_id, dataset_id;  /* pointers to the HDF5 file and input dataset */
  float* tmpIN = NULL;        /* pointer to the inputs buffer */

  file_id = my_h5f_open(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset_id = my_h5d_open2(file_id, mtx_name, H5P_DEFAULT);
  *N_in = get_matrix_size(dataset_id);
  tmpIN = my_malloc_float(sizeof(float) * (*N_in)); 
  my_h5d_read(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmpIN);
  my_h5d_close(dataset_id);
  my_h5f_close(file_id);
  
  return tmpIN;
  
}

/**
 * Load the AI model's input vectors from an HDF5 file.
 *
 * Arguments:
 *   - file_loc: The path to the data files
 *   - n_samples: Pointer to a buffer in which to save the number of loaded input vectors
 *   - n_inputs: The number of features in each input vector.
 *
 * Returns:
 *   - The buffer containing the input vectors.
 */
float* load_inputs(char* file_loc, int* n_samples, int n_inputs) {

  float* buffer = NULL;            /* pointer to the buffer containing the inputs */
  unsigned long buf_sz = 0;        /* the number of number in the buffer */
  
  buffer = load_matrix(file_loc, INPUTS_KEY, &buf_sz);
  *n_samples = (int) buf_sz / n_inputs; 
  
  return buffer;
  
}

/**
 * Load the AI model's auxiliary inputs vectors from an HDF5 file.
 *
 * Arguments:
 *   - file_loc: The path to the data files
 *   - n_samples: Pointer to a buffer in which to save the number of loaded input vectors
 *   - n_aux_inputs: The number of features in the sample
 *
 * Returns:
 *   - The buffer containing the 3D coordinates.
 */
float* load_aux_inputs(char* file_loc, int* n_samples, int n_aux_inputs) {

  float* buffer = NULL;            /* pointer to the buffer containing the inputs */
  unsigned long buf_sz = 0;        /* the number of number in the buffer */
  
  buffer = load_matrix(file_loc, AUX_INPUTS_KEY, &buf_sz);
  *n_samples = (int) buf_sz / n_aux_inputs;
  
  return buffer;
  
}

/**
 * Extract a vector of input features from the buffer containing all
 * input feature vectors.
 * 
 * Arguments:
 *   - input_buffer: The buffer containing all input feature vectors
 *   - sample_vec: Pointer to the buffer where the a single sample's input
 *                 features should be saved.
 *   - n_inputs: The number of features in the sample
 *   - sample_id: The index of the sample to read from the buffer
 *
 * Returns:
 *   - Nothing.
 */
void get_sample(float* input_buffer, float* sample_vec, int n_inputs, int sample_id) {

  int ii = 0;
  
  for (ii = 0; ii < n_inputs; ii++) 
    sample_vec[ii] = input_buffer[n_inputs*sample_id+ii];

  sample_vec[n_inputs] = 1.0; /* Add a constant 1 for the bias parameter. */
  
}

