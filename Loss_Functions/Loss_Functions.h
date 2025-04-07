/**
 * @file Loss_Functions.h
 * @brief Provides loss function implementations for neural network training
 */

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H

#include <MatrixLibrary.h>
#include <cassert>

// Forward declarations
class Matrix;
namespace Matrix_Checks {
    bool Matrix_Can_AddOrSubtract(const Matrix& Result, const Matrix& A, const Matrix& B);
}

/**
 * @enum LossFunction
 * @brief Enumeration of supported loss functions for neural network training
 */
enum class LossFunction {
    MSE_LOSS,            ///< Mean Squared Error Loss
    HUBER_LOSS,          ///< Huber Loss (robust to outliers)
    CROSS_ENTROPY_LOSS,  ///< Binary Cross-Entropy Loss (for classification)
    MAE_LOSS             ///< Mean Absolute Error Loss (L1 Loss)
};

// Function declarations

/**
 * @brief Calculates Mean Squared Error loss between predicted and target values
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @return The MSE loss value
 */
float Mean_Squared_Error_Loss(Matrix &predicted, Matrix &target);

/**
 * @brief Calculates Mean Absolute Error (L1) loss between predicted and target values
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @return The MAE loss value
 */
float Mean_Absolute_Error_Loss(Matrix &predicted, Matrix &target);

/**
 * @brief Calculates Huber loss between predicted and target values
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param delta Threshold parameter for Huber loss
 * @return The Huber loss value
 */
float Huber_Loss(Matrix &predicted, Matrix &target, float delta);

/**
 * @brief Calculates Binary Cross-Entropy loss between predicted and target values
 * @param predicted Matrix containing model predictions (values should be between 0 and 1)
 * @param target Matrix containing target values (binary values 0 or 1)
 * @return The Binary Cross-Entropy loss value
 */
float Binary_Cross_Entropy_Loss(Matrix &predicted, Matrix &target);

/**
 * @brief Generic loss function dispatcher that calls the appropriate loss function
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param loss_function The type of loss function to use
 * @return The calculated loss value
 * @throw std::invalid_argument if an unsupported loss function is specified
 */
float Calculate_Loss(Matrix &predicted, Matrix &target, LossFunction loss_function);

/**
 * @brief Overloaded loss function dispatcher that includes a delta parameter (for Huber loss)
 * @param predicted Matrix containing model predictions
 * @param target Matrix containing target values
 * @param delta Parameter for loss functions that require it (e.g., Huber loss)
 * @param loss_function The type of loss function to use
 * @return The calculated loss value
 * @throw std::invalid_argument if an unsupported loss function is specified
 */
float Calculate_Loss(Matrix &predicted, Matrix &target, float delta, LossFunction loss_function);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LOSS_FUNCTIONS_H