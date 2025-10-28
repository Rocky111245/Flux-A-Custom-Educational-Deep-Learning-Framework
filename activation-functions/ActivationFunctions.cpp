// Activation_Functions.cpp
// Created by Rocky170 on 7/23/2025.

#include "ActivationFunctions.h"
#include <cmath>
#include <algorithm>
#include "tensor-library/TensorLibrary.h"

void ReLU(Tensor& dst, const Tensor& src) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                dst(r, c, d) = std::max(0.0f, src(r, c, d));
            }
        }
    }
}

void GELU(Tensor& dst, const Tensor& src) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;

    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                float x = src(r, c, d);
                float tanh_input = sqrt_2_over_pi * (x + coeff * x * x * x);
                dst(r, c, d) = 0.5f * x * (1.0f + std::tanh(tanh_input));
            }
        }
    }
}

void Leaky_ReLU(Tensor& dst, const Tensor& src, float alpha) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                float x = src(r, c, d);
                dst(r, c, d) = x > 0.0f ? x : alpha * x;
            }
        }
    }
}

void Swish(Tensor& dst, const Tensor& src) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                float x = src(r, c, d);
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                dst(r, c, d) = x * sigmoid_x;
            }
        }
    }
}

void Tanh(Tensor& dst, const Tensor& src) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                dst(r, c, d) = std::tanh(src(r, c, d));
            }
        }
    }
}

void Sigmoid(Tensor& dst, const Tensor& src) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                dst(r, c, d) = 1.0f / (1.0f + std::exp(-src(r, c, d)));
            }
        }
    }
}

void Linear(Tensor& dst, const Tensor& src) {
    for (int d = 0; d < src.depth(); ++d) {
        for (int r = 0; r < src.rows(); ++r) {
            for (int c = 0; c < src.columns(); ++c) {
                dst(r, c, d) = src(r, c, d);
            }
        }
    }
}
