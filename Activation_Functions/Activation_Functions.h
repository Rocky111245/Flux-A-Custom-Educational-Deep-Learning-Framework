//
// Created by Rocky170 on 7/23/2025.
//

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H


class Tensor;

void ReLU(Tensor& dst, const Tensor& src);
void GELU(Tensor& dst, const Tensor& src);
void Leaky_ReLU(Tensor& dst, const Tensor& src, float alpha = 0.01f);
void Swish(Tensor& dst, const Tensor& src);
void Tanh(Tensor& dst, const Tensor& src);
void Sigmoid(Tensor& dst, const Tensor& src);
void Linear(Tensor& dst, const Tensor& src);

#endif //ACTIVATION_FUNCTIONS_H
