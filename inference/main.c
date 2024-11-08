
#include <time.h>
#include "model.h"
#include "hdf5_read.h"

#define TRUE 1

/* The location of the data files */
char* file_loc = "/home/bbooth/scratch/FAIR/datasets/test_data_A.h5";

/**
 * Process a layer of monitoring data from a print job. These data are
 * provided along with the corresponding x,y coordinates of the laser at the
 * time that the data was collected. Save the output to the given buffer and
 * compute the average computation time for a data sample in the layer.
 *
 * Arguments:
 *   - inputs_buf: buffer containing all of the input vectors
 *   - coords_buf: buffer containing the of the laser x,y,z coordinates
 *   - output_buf: buffer in which to save the outputs of the AI model and the
 *                 corresponding x,y coordinates.
 *   - n_samples: The number of samples in each buffer.
 *
 * Returns:
 *   - The average computation time needed to process a data sample (in microseconds)
 */
float process_layer(float* inputs_buf, float* aux_in_buf, float* output_buf, int n_samples) {

  float sample[INPUT_SZ+1] = {1.0}; /* Buffer for the current sample (+1 for bias) */
  float preds[OUTPUT_SZ] = {0.0};   /* Buffer for the current outputs */
  float aux_in[AUX_IN_SZ+1] = {1.0}; /* Laser coords and layer number for current sample */
  int ii = 0;                       /* Index over the processing of samples */
  clock_t begin, end;               /* Timing variables for looging algorithm speed */
  float avg_comp_time = 0.0;        /* Average computation time per sample */
  
  /* tic */
  begin = clock();
  
  /* Loop over all samples */
  for (ii = 0; ii < n_samples; ii++) {

    /* Get sample */
    get_sample(inputs_buf, sample, INPUT_SZ, ii);
    get_sample(aux_in_buf, aux_in, AUX_IN_SZ, ii);

    /* Process sample */
    infer(sample, aux_in, preds);

    /* Log outputs */
    output_buf[ii*OUTPUT_SZ] = preds[0];    /* predicted melt pool intensity */
    output_buf[ii*OUTPUT_SZ+1] = preds[1];  /* predicted melt pool size */
    output_buf[ii*OUTPUT_SZ+2] = preds[2];  /* predicted number of spatters */
      
  }

  /* toc */
  end = clock();

  /* Calculate the amount of time per sample in microseconds. */
  avg_comp_time = (float) (end - begin) / (CLOCKS_PER_SEC);
  avg_comp_time = avg_comp_time / n_samples;
  avg_comp_time = avg_comp_time * 1e6;

  return avg_comp_time;
  
}

/* Entry point of the program */
int main(int argc, char *argv[]) {

  float comp_time = 0.0; /* The computation time for the dataset */

  int n_samples = 0;              /* The number of samples in the dataset */
  float* inputs_buf = NULL;       /* Pointer to input vectors buffer */
  float* aux_in_buf = NULL;       /* Pointer to auxiliary inputs buffer */
  float* output_buf = NULL;       /* Pointer to output buffer to be saved */
 
  /* Load the dataset values from file */
  /* Note that these buffers need to be freed after each iteration */
  inputs_buf = load_inputs(file_loc, &n_samples, INPUT_SZ);
  aux_in_buf = load_aux_inputs(file_loc, &n_samples, AUX_IN_SZ);
  output_buf = my_malloc_float(OUTPUT_SZ*n_samples*sizeof(float));
    
  /* Process the data */
  comp_time = process_layer(inputs_buf, aux_in_buf, output_buf, n_samples);
  printf("Average Speed (in microseconds): %g\n", comp_time);
  
  /* Free up any malloc'ed memory */
  free(inputs_buf);
  free(aux_in_buf);
  free(output_buf);

  printf("Exiting cleanly...\n");
  
}
