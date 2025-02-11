
#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H

#include <MatrixLibrary.h>
enum class LossFunction {
    MSE_LOSS,
    HUBER_LOSS,
    CROSS_ENTROPY_LOSS,
};
float Calculate_Loss(Matrix &predicted, Matrix &target, LossFunction loss_function);
float Calculate_Loss(Matrix &predicted, Matrix &target, float delta, LossFunction loss_function);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H
