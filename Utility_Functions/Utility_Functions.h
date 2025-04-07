/**
 * @file Utility_Functions.h
 * @brief Provides utility functions for debugging and string conversion in the neural network framework
 * @author Rakib
 * @date 2025-03-28
 */

#ifndef NEWFRAMEWORK_UTILITY_FUNCTIONS_H
#define NEWFRAMEWORK_UTILITY_FUNCTIONS_H

#include <string>

// Forward declarations
class Train_Block_by_Backpropagation;
class Matrix;
enum class ActivationType;

/**
 * @brief Prints a matrix with detailed formatting for debugging purposes
 *
 * Displays matrix dimensions, name, and optionally its full contents with
 * configurable precision. Helps visualize matrix values during network
 * training and debugging.
 *
 * @param matrix The matrix to print
 * @param matrix_name Name of the matrix (e.g., "Weights", "Bias")
 * @param layer_number Layer number (-1 for matrices not associated with a specific layer)
 * @param precision Number of decimal places to show (default: 4)
 * @param show_full Whether to show the full matrix contents or just dimensions (default: true)
 */
void Matrix_Debug_Print(const Matrix& matrix, const std::string& matrix_name,
                        int layer_number, int precision = 4, bool show_full = true);

/**
 * @brief Converts an activation function type to its string representation
 *
 * Provides a human-readable name for each activation function type,
 * useful for logging, UI display, and debugging.
 *
 * @param type The activation function type to convert
 * @return String representation of the activation type (e.g., "ReLU", "Sigmoid")
 */
std::string GetActivationTypeString(ActivationType type);

#endif //NEWFRAMEWORK_UTILITY_FUNCTIONS_H