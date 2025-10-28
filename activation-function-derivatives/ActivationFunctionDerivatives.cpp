//
// Created by Rocky170 on 7/22/2025.
//

#include "ActivationFunctionDerivatives.h"
#include "tensor-library/TensorLibrary.h"

Tensor Sigmoid_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                const float sigmoid_x = 1.0f / (1.0f + std::exp(-input(r, c, d)));
                result(r, c, d) = sigmoid_x * (1.0f - sigmoid_x);
            }
        }
    }
    return result;
}


Tensor Relu_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                result(r, c, d) = input(r, c, d) > 0.0f ? 1.0f : 0.0f;
            }
        }
    }
    return result;
}


Tensor Gelu_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                float x = input(r, c, d);
                float tanh_input = sqrt_2_over_pi * (x + coeff * x * x * x);
                float tanh_val = std::tanh(tanh_input);
                float sech2_val = 1.0f - tanh_val * tanh_val;
                float dtanh_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);

                result(r, c, d) = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2_val * dtanh_dx;
            }
        }
    }
    return result;
}

Tensor Leaky_Relu_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());
    const float alpha = 0.01f;

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                result(r, c, d) = input(r, c, d) > 0.0f ? 1.0f : alpha;
            }
        }
    }
    return result;
}


Tensor Swish_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                float x = input(r, c, d);
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                result(r, c, d) = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
            }
        }
    }
    return result;
}

Tensor Tanh_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                float tanh_x = std::tanh(input(r, c, d));
                result(r, c, d) = 1.0f - tanh_x * tanh_x;
            }
        }
    }
    return result;
}


Tensor Linear_Derivative(const Tensor &input) {
    Tensor result(input.rows(), input.columns(), input.depth());

    for (int d = 0; d < input.depth(); ++d) {
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.columns(); ++c) {
                result(r, c, d) = 1.0f;
            }
        }
    }
    return result;
}
