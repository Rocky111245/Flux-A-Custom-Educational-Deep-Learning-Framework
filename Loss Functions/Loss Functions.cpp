
#include "Loss Functions.h"



// Mean Squared Error Loss
float Mean_Squared_Error_Loss(Matrix &predicted, Matrix &target) {
    Matrix diff = Matrix_AutoCreate(predicted, target);
    Matrix_Subtract(diff, predicted, target);  // predicted - target
    int num_samples = predicted.rows(); // Use matrix size directly

    Matrix_Power(diff, 2.0f);  // Square all elements
    float MSE=Matrix_Sum_All_Elements(diff) / num_samples;
    return MSE;
}


// Huber Loss
float Huber_Loss(Matrix &predicted, Matrix &target,float delta) {
    Matrix diff = Matrix_AutoCreate(predicted, target);
    Matrix_Subtract(diff, predicted, target);  // predicted - target
    int num_samples = predicted.rows(); // Use matrix size directly

    float total_loss = 0.0f;
    for(int i = 0; i < diff.rows(); ++i) {
        float error = diff(i, 0);
        float abs_error = std::abs(error);

        if(abs_error <= delta) {
            total_loss += 0.5f * error * error;
        } else {
            total_loss += delta * (abs_error - 0.5f * delta);
        }

    }
    float huber_loss_error=total_loss / num_samples;
    return huber_loss_error;
}


// Cross-Entropy Loss/Binary Cross Entropy Loss. Please note that BCE is used with sigmoid and CE is used with softmax usually.
float Binary_Cross_Entropy_Loss(Matrix &predicted, Matrix &target) {
    constexpr float EPSILON = 1e-7f;  // Avoid log(0)
    float total_loss = 0.0f;
    int num_samples = predicted.rows(); // Use matrix size directly

    for (int i = 0; i < num_samples; ++i) {
        float y = target(i, 0);
        float y_hat = std::clamp(predicted(i, 0), EPSILON, 1.0f - EPSILON);
        total_loss += y * std::log(y_hat) + (1.0f - y) * std::log(1.0f - y_hat);
    }

    return -total_loss / num_samples;  // Negating the loss
}

float Calculate_Loss(Matrix &predicted, Matrix &target, LossFunction loss_function) {
    switch (loss_function) {
        case LossFunction::MSE_LOSS:
            return Mean_Squared_Error_Loss(predicted, target);

        case LossFunction::CROSS_ENTROPY_LOSS:
            return Binary_Cross_Entropy_Loss(predicted, target);

        default:
            throw std::invalid_argument("Invalid loss function type for this overload.");
    }
}

// Overloaded function for Huber Loss with delta parameter
float Calculate_Loss(Matrix &predicted, Matrix &target, float delta, LossFunction loss_function) {
    if (loss_function == LossFunction::HUBER_LOSS) {
        return Huber_Loss(predicted, target, delta);
    } else {
        throw std::invalid_argument("Invalid loss function type for this overload.");
    }
}




