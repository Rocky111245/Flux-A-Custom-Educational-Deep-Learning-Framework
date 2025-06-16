
#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_POOLING_LAYERS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_POOLING_LAYERS_H

/*
* Mechanical Understanding Pooling Layers (ONLY 1 POOLING WINDOW)
* ---------------------------------------------------------------
*
* 1. Input: 3D tensor [height × width × depth]
*    - Depth represents feature maps from previous layers
*
* 2. Pooling Window: 2D spatial region [size × size]
*    - Defines the area to pool from (e.g., 2×2 or 3×3)
*    - No learnable parameters or weights
*    - Applied independently to each channel
*
* 3. Pooling operations: Pooling window slides across each channel of the input tensor
*    - MAX POOL: Gets maximum value within each window (stores indices for backpropagation)
*    - AVERAGE POOL: Averages values within each window (no indices storage required)
*
* 4. Output: 3D tensor ['downsampled height' × 'downsampled width' × same depth as input]
*    - Spatial dimensions reduced based on pooling window size and stride
*    - Channel dimension remains unchanged
*    - Becomes input to the next layer
*/



#include "Utility_Migratory_Functions/Utility_Migratory_Functions.h"
#include "Tensor_Library/Tensor_Library.h"
#include "MatrixLibrary.h"
#include "Padding Functions/Tensor_Padding.h"
#include <vector>

enum PoolingStrategy{
    MAX_POOL,
    AVERAGE_POOL
};


class Pooling_Layer {
public:
    Pooling_Layer(const Tensor &input_tensor,int pool_window_size, int stride, PoolingStrategy pooling_strategy, PaddingStrategy padding_strategy=SAME)
    :input_tensor_(input_tensor), pooling_strategy_(pooling_strategy), pool_window_size_(pool_window_size), stride_(stride), padding_strategy_(padding_strategy)
    {
        //Fixes output dimensions and pads the input tensor
        Initialize_Tensors();
    }

    // Methods for pooling operations

private:
    // Configuration
    int pool_window_size_;
    int stride_;

    // Tensors used during computation
    Tensor input_tensor_;
    Tensor padded_input_tensor_;
    Tensor output_tensor_;
    // For max pooling backpropagation
    Tensor max_pool_indices_tensor_;             // To store positions of max values. Each row of tensor store indices of (row,column). Each this tensor also stores the indices to respective channels.

    Tensor dL_dy_;                               // Backpropagated gradient from 'upper' layer or upstream gradient.
    Tensor dL_dx_;                              // Backpropagated gradient to 'lower' layer or downstream gradient. Gradient w.r.t. input (we want to compute this).
    PoolingStrategy  pooling_strategy_;
    PaddingStrategy  padding_strategy_;

    void Apply_Padding();

    void Max_Pooling();
    void Average_Pooling();
    Tensor Get_Max_Pool_Tensor_Indices() const { return max_pool_indices_tensor_;}
    void Initialize_Tensors();

    void Backpropagate(Tensor &dL_dx, Tensor &dL_dy);
};




#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_POOLING_LAYERS_H
