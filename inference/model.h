
#include <math.h> /* for the tanh function */ 

#ifndef _MODEL_H_
#define _MODEL_H_

#define INPUT_SZ 40 /* Number of input features */
#define AUX_IN_SZ 6 /* Number of auxiliary features */
#define HIDDEN_SZ 20 /* Number of hidden nodes */ 
#define OUTPUT_SZ 3 /* Number of predicted outputs */ 

#define DELTA_T_ID 5 /* Auxiliary feature index for the delta_T feature.
		      * Used for the masking procedure in the AWN.
		      */

/* External declarations of model parameters */
extern const float layerAUX [(INPUT_SZ+1)*HIDDEN_SZ][AUX_IN_SZ+1];
extern const float layerOUT [OUTPUT_SZ][HIDDEN_SZ+1];
extern const float normAUX [AUX_IN_SZ][2];
extern const float normINP [INPUT_SZ][2];

/* Functions used to infer outputs from the learned neural network. */
void infer(float* inputs, float* aux_in, float* outputs); 

#endif
