//
// Created by rakib on 28/3/2025.
//

#ifndef NEWFRAMEWORK_UTILITY_FUNCTIONS_H
#define NEWFRAMEWORK_UTILITY_FUNCTIONS_H

#include <string>

// Forward declaration of the Train_Block_by_Backpropagation class
class Train_Block_by_Backpropagation;
class Matrix;

// Forward declaration of ActivationType enum
enum class ActivationType;

// Function declarations
void DisplayNeuralLayerMatrices(const Train_Block_by_Backpropagation& trained_block);
void DisplayBackpropagationGradients(const Train_Block_by_Backpropagation& trained_block);
void Debug_Matrix(const Matrix& matrix, const std::string& name, int precision = 4, int max_rows = 0, int max_cols = 0);
std::string GetActivationTypeString(ActivationType type);

#endif //NEWFRAMEWORK_UTILITY_FUNCTIONS_H