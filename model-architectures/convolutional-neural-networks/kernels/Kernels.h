// Kernels.h
// Defines the Kernel class for convolutional operations. It is a wrapper over the Tensor class.
// Author: Rakib
// Date: 2025-04-24

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_KERNELS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_KERNELS_H

#include "tensor-library/TensorLibrary.h"


class Kernel : public Tensor {
public:
    Kernel(int size, int channels) : Tensor(size, size, channels) {}
    void Kernel_Xavier_Uniform(int number_of_kernels);
    Tensor Flatten_Kernel() const;

    int size() const noexcept { return rows(); }
    int channels() const noexcept { return depth(); }



};


#endif // _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_KERNELS_H
