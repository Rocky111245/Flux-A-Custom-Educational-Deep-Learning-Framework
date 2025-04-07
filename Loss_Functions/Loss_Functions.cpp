/**
 * @file Loss_Functions.cpp
 * @brief Implementation of loss functions for neural network training
 */

#include "Loss_Functions.h"
#include "Matrix_Dimension_Checks/Matrix_Assertion_Checks.h"
#include <algorithm> // For std::clamp
#include <cmath>     // For std::log, std::exp, std::abs

/**
 * @brief Calculates Mean Squared Error loss between predicted and target values
 *
 * MSE = (1/n) * Σ(predicted - target)²
 *
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @return The MSE loss value
 */
float Mean_Squared_Error_Loss(Matrix &predicted, Matrix &target) {
    // Create a matrix to hold the differences
    Matrix diff(predicted.rows(), predicted.columns(), 0.0f);

    // Ensure matrices have compatible dimensions
    assert(Matrix_Can_AddOrSubtract(diff, predicted, target));

    // Calculate differences (predicted - target)
    Matrix_Subtract(diff, predicted, target);

    // Number of samples
    int num_samples = predicted.rows();

    // Square the differences
    Matrix_Power(diff, 2.0f);

    // Calculate mean of squared differences
    float MSE = Matrix_Sum_All_Elements(diff) / num_samples;

    return MSE;
}

/**
 * @brief Calculates Mean Absolute Error (L1) loss between predicted and target values
 *
 * MAE = (1/n) * Σ|predicted - target|
 *
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @return The MAE loss value
 */
float Mean_Absolute_Error_Loss(Matrix &predicted, Matrix &target) {
    // Create a matrix for the differences with appropriate dimensions
    Matrix diff = Matrix_AutoCreate(predicted, target);

    // Ensure matrices have compatible dimensions
    assert(Matrix_Can_AddOrSubtract(diff, predicted, target));

    // Calculate differences (predicted - target)
    Matrix_Subtract(diff, predicted, target);

    // Number of samples
    int num_samples = predicted.rows();

    // Take absolute values of the differences
    Matrix abs_diff = Matrix_AutoCreate(diff, diff);
    Matrix_Absolute(abs_diff, diff);

    // Calculate mean of absolute differences
    float mae_loss = Matrix_Sum_All_Elements(abs_diff) / num_samples;

    return mae_loss;
}

/**
 * @brief Calculates Huber loss between predicted and target values
 *
 * Huber loss combines MSE and MAE to be robust to outliers:
 * L(error) = 0.5 * error² if |error| ≤ delta
 *          = delta * (|error| - 0.5 * delta) otherwise
 *
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param delta Threshold parameter that determines the transition from quadratic to linear
 * @return The Huber loss value
 */
float Huber_Loss(Matrix &predicted, Matrix &target, float delta) {
    // Create a matrix for the differences with appropriate dimensions
    Matrix diff = Matrix_AutoCreate(predicted, target);

    // Ensure matrices have compatible dimensions
    assert(Matrix_Can_AddOrSubtract(diff, predicted, target));

    // Calculate differences (predicted - target)
    Matrix_Subtract(diff, predicted, target);

    // Number of samples
    int num_samples = predicted.rows();

    // Calculate Huber loss
    float total_loss = 0.0f;
    for(int i = 0; i < diff.rows(); ++i) {
        float error = diff(i, 0);
        float abs_error = std::abs(error);

        if(abs_error <= delta) {
            // Quadratic part for small errors
            total_loss += 0.5f * error * error;
        } else {
            // Linear part for large errors
            total_loss += delta * (abs_error - 0.5f * delta);
        }
    }

    // Calculate mean loss
    float huber_loss_error = total_loss / num_samples;

    return huber_loss_error;
}

/**
 * @brief Calculates Binary Cross-Entropy loss between predicted and target values
 *
 * BCE = -(1/n) * Σ[target * log(predicted) + (1 - target) * log(1 - predicted)]
 *
 * @param predicted Matrix containing model predictions (values between 0 and 1)
 * @param target Matrix containing target values (binary values 0 or 1)
 * @return The Binary Cross-Entropy loss value
 */
float Binary_Cross_Entropy_Loss(Matrix &predicted, Matrix &target) {
    constexpr float EPSILON = 1e-7f;  // Small constant to avoid log(0)
    float total_loss = 0.0f;

    // Number of samples
    int num_samples = predicted.rows();

    // Calculate BCE loss
    for (int i = 0; i < num_samples; ++i) {
        float y = target(i, 0);
        // Clamp predicted values to avoid numerical instability
        float y_hat = std::clamp(predicted(i, 0), EPSILON, 1.0f - EPSILON);

        // BCE formula
        total_loss += y * std::log(y_hat) + (1.0f - y) * std::log(1.0f - y_hat);
    }

    // Return negative mean (cross-entropy is a minimization objective)
    return -total_loss / num_samples;
}

/**
 * @brief Generic loss function dispatcher that calls the appropriate loss function
 *
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param loss_function The type of loss function to use
 * @return The calculated loss value
 * @throw std::invalid_argument if an unsupported loss function is specified
 */
float Calculate_Loss(Matrix &predicted, Matrix &target, LossFunction loss_function) {
    switch (loss_function) {
        case LossFunction::MSE_LOSS:
            return Mean_Squared_Error_Loss(predicted, target);

        case LossFunction::CROSS_ENTROPY_LOSS:
            return Binary_Cross_Entropy_Loss(predicted, target);

        case LossFunction::MAE_LOSS:
            return Mean_Absolute_Error_Loss(predicted, target);

        default:
            throw std::invalid_argument("Invalid loss function type for this overload.");
    }
}

/**
 * @brief Overloaded loss function dispatcher that includes a delta parameter (for Huber loss)
 *
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param delta Parameter for loss functions that require it (e.g., Huber loss)
 * @param loss_function The type of loss function to use
 * @return The calculated loss value
 * @throw std::invalid_argument if an unsupported loss function is specified
 */
float Calculate_Loss(Matrix &predicted, Matrix &target, float delta, LossFunction loss_function) {
    if (loss_function == LossFunction::HUBER_LOSS) {
        return Huber_Loss(predicted, target, delta);
    } else {
        throw std::invalid_argument("Invalid loss function type for this overload.");
    }
}