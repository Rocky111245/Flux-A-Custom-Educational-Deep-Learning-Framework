/**
 * @file Utility_Functions.cpp
 * @brief Implementation of utility functions for debugging and string conversion
 * @author Rakib
 * @date 2025-03-28
 */

#include <sstream>
#include <iomanip>
#include "Utility_Functions.h"
#include "Multi Layer Perceptron/Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include "Multi Layer Perceptron/Neural_Layers/Neural_Layer_Skeleton.h" // For ActivationType

void Matrix_Debug_Print(const Matrix& matrix, const std::string& matrix_name,
                        int layer_number, int precision, bool show_full) {
    // Print separator line for visual clarity
    std::cout << "-------------------------------------" << std::endl;

    // Print header with matrix name and layer info
    if (layer_number >= 0) {
        std::cout << "Layer " << layer_number << " - ";
    }
    std::cout << matrix_name << " [" << matrix.rows() << " X " << matrix.columns() << "]" << std::endl;

    // Print matrix contents if requested
    if (show_full) {
        std::cout << "-------------------------------------" << std::endl;
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.columns(); ++j) {
                // Format each element with consistent width and precision
                std::cout << std::fixed << std::setprecision(precision)
                          << std::setw(precision + 3) << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Print closing separator line
    std::cout << "-------------------------------------" << std::endl;
}

std::string GetActivationTypeString(ActivationType type) {
    // Map each activation type to its string representation
    switch (type) {
        case ActivationType::RELU:       return "ReLU";
        case ActivationType::SIGMOID:    return "Sigmoid";
        case ActivationType::TANH:       return "Tanh";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::SWISH:      return "Swish";
        case ActivationType::LINEAR:     return "Linear";
        default:                         return "Unknown";
    }
}