//
// Created by rakib on 17/5/2025.
//

/*
* Understanding Convolutional Neural Network Operations
* -----------------------------------------------------
*
* 1. Input: 3D tensor [height × width × depth]
*    - Depth represents channels (RGB) or feature maps from previous layers
*
* 2. Kernels: 3D tensors [filter size × filter size × depth]
*    - Must match input depth
*    - Square-shaped, multiple used per layer for different feature detection
*
* 3. Convolution: Kernels slide across input tensor
*    - Element-wise multiplication and summation across depth
*    - Each kernel produces one 2D feature map
*
* 4. Bias: Scalar parameter added to each 2D feature map
*    - Applied element-wise after convolution
*    - Adjusts activation thresholds independently
*
* 5. Activation: Non-linear functions applied element-wise
*    - Applied to each value in feature maps after bias addition
*    - Introduces non-linearity for learning complex patterns
*
* 6. Output: 3D tensor [height' × width' × number of kernels]
*    - Formed by stacking all 2D feature maps along depth dimension
*    - Becomes input to next layer
*/


#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_CONVOLUTIONAL_LAYER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_CONVOLUTIONAL_LAYER_H

#include "MatrixLibrary.h"
#include "tensor-library/TensorLibrary.h"
#include "../kernels/Kernels.h"
#include "../padding-functions/TensorPadding.h"

// Define activation types directly here instead of importing from Neural_Layer_Skeleton
enum class ConvActivationType {
    LINEAR,
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
};

class Convolutional_Layer_Skeleton {
public:
    // Constructor for convolutional layer
    Convolutional_Layer_Skeleton(Tensor& input_tensor, int number_of_kernels, ConvActivationType activation_type,
                                 PaddingStrategy padding_strategy = PaddingStrategy::VALID, int stride = 1);

    // Destructor
    ~Convolutional_Layer_Skeleton() = default;





private:

    // Member variables
    Tensor input_tensor_;                           //input tensor present in every layer in this architecture, a universal truth.
    Tensor padded_input_tensor_;                    //padded tensor constructed from the input tensor
    Tensor  im2rows_tensor_;                        //im2rows constructed from padded_input_tensor
    std::vector<Kernel> kernels_vectors_;           // all kernels are stored in a vector. Kernel values are the weights we commonly understand in neural networks
    std::vector<Tensor> flattened_kernels_vectors_; //all flattened kernels are stored in a vector in a 1D form
    Tensor bias_tensor_;                            //bias tensor added to pre-activated outputs. Each tensor channel for each kernel.
    Tensor pre_activation_tensor_;                  //pre_activation_tensors for each kernel. Each tensor channel for each kernel.
    Tensor post_activation_tensor_;                 //post_activation_tensors for each activated kernel output. Each tensor channel for each kernel.
    Tensor output_tensor_;                          //each layer must produce an output tensor, to feed to the next layer.
    ConvActivationType activation_type_;
    PaddingStrategy padding_strategy_;

    int number_of_kernels_;                         // number of kernels we define
    int stride_;                                    // Step size for convolution
    int padding_;                                   // Padding size around input. If padding is wrong (kernel goes out of bounds), mathematically calculated padding will be added


    void Initialize_All_Tensors();
};




/*------------------------DESIGN DECISIONS---------------------
 * It would have been possible to make multiple kernels just form a large kernel,like if we wanted 5 kernels,each of depth 2,we could have a big kernel of 10 channels,
 * where after every 2 tensor blocks,we get the start of a different kernel. However, this design choice was not used due to readability issues and a vector design approach
 * was deemed cleaner. The user comprehension is of utmost importance.
 *
 *
 *
 *
 */

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_CONVOLUTIONAL_LAYER_H
