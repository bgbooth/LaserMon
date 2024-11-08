#include "model.h"

/* Buffers for neural network computations */
float layerINP [HIDDEN_SZ][INPUT_SZ+1];
float hidden_vals [HIDDEN_SZ+1] = {1.0};

/**
 * Rectified linear unit function. Returns the input value unless
 * it is negative. In that case, it returns zero.
 *
 * Arguments:
 *   - input: The value to send through the ReLU function.
 *
 * Returns:
 *   - The computed output of the ReLU function.
 */
float relu(const float input) {
  return input < 0.0 ? 0.0 : input;
}

/**
 * Normalize an input feature vector to its z-scores.
 *
 * Arguments:
 *   - inputs: array of input features of size INPUT_SZ+1 to be 
 *             normalized.
 *
 * Returns:
 *   - Nothing. Updates 'inputs' with normalized values.
 */
void normalize_inputs(float* inputs) {

  int ii = 0;

  for (ii = 0; ii < INPUT_SZ; ii++)
    inputs[ii] = (inputs[ii] - normINP[ii][0]) / normINP[ii][1];
  
}

/**
 * Normalize an auxiliary input feature vector to its z-scores.
 *
 * Arguments:
 *   - aux_inputs: array of auxiliary input features of size 
 *                 AUX_IN_SZ+1 to be normalized.
 *
 * Returns:
 *   - Nothing. Updates 'aux_inputs' with normalized values.
 */
void normalize_aux_inputs(float* aux_inputs) {

  int ii = 0;

  for (ii = 0; ii < AUX_IN_SZ; ii++)
    aux_inputs[ii] = (aux_inputs[ii] - normAUX[ii][0]) / normAUX[ii][1];
  
}

/**
 * Compute the value of the given hidden node using the input
 * features provided.
 *
 * Arguments:
 *   - input_id: the index of the input feature whose layer weights are to be 
 *               computed.
 *   - hidden_id: the index of the hidden node whose layer weights are to be
 *                computed.
 *   - inputs: array of input features of size INPUT_SZ+1 to be used
 *             to compute the value of the hidden node.
 *
 * Returns:
 *   - Nothing.
 */
void get_input_layer_params(const int input_id, const int hidden_id,
			    const float* aux_inputs) {

  int ii = 0;
  layerINP[hidden_id][input_id] = 0.0;

  if ((input_id >= 2 * aux_inputs[DELTA_T_ID]) && (input_id < INPUT_SZ))
    return;
  
  for (ii = 0; ii < AUX_IN_SZ+1; ii++) 
    layerINP[hidden_id][input_id] += layerAUX[HIDDEN_SZ*input_id+hidden_id][ii] * aux_inputs[ii];
  
}

/**
 * Compute the value of the given hidden node using the input
 * features provided.
 *
 * Arguments:
 *   - node_id: the index of the hidden node whose value is to be 
 *              computed.
 *   - inputs: array of input features of size INPUT_SZ+1 to be used
 *             to compute the value of the hidden node.
 *
 * Returns:
 *   - Nothing.
 */
void get_hidden_node_value(const int node_id, const float* inputs) {

  int ii = 0;
  hidden_vals[node_id] = 0.0;

  for (ii = 0; ii < INPUT_SZ+1; ii++) 
    hidden_vals[node_id] += layerINP[node_id][ii] * inputs[ii];

  hidden_vals[node_id] = (float) tanh(hidden_vals[node_id]);
  
}

/**
 * Compute the value of the given output node using the current values
 * in the global hidden node array.
 *
 * Arguments:
 *   - output_id: the index of the output whose value is to be 
 *                computed.
 *
 * Returns:
 *   - The computed value of the output node.
 */
float get_output_value(const int output_id) {

  int ii = 0;
  float output_val = 0.0;

  for (ii = 0; ii < HIDDEN_SZ+1; ii++)
    output_val += layerOUT[output_id][ii] * hidden_vals[ii];

  return relu(output_val);
  
}

/**
 * Perform the inference of the melt pool state from the provided
 * photodiode and image feature inputs. 
 *
 * Arguments:
 *   - inputs: array of input features of size INPUT_SZ+1 to be used
 *             to compute the value of the hidden node.
 *   - outputs: array of model outputs of size OUTPUT_SZ. Used to 
 *              return the values predicted by the model.
 *
 * Returns:
 *   - Nothing.
 */
void infer(float* inputs, float* aux_in, float* outputs) {

  int ii = 0;
  int jj = 0;

  normalize_inputs(inputs);
  normalize_aux_inputs(aux_in);

  for (ii = 0; ii < INPUT_SZ+1; ii++)
    for (jj = 0; jj < HIDDEN_SZ; jj++)
      get_input_layer_params(ii, jj, aux_in);
      
  for (ii = 0; ii < HIDDEN_SZ; ii++)
    get_hidden_node_value(ii, inputs);

  for (ii = 0; ii < OUTPUT_SZ; ii++)
    outputs[ii] = get_output_value(ii);
}
