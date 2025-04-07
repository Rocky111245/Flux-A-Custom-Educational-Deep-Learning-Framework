/**
 * @file Matrix_Assertion_Checks.h
 * @brief Provides utility functions for verifying matrix dimension compatibility
 * @author rakib
 * @date 25/3/2025
 */

#ifndef NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H
#define NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H

#include <iostream>
#include <MatrixLibrary.h>

// Forward declarations
class Matrix;

/**
 * @brief Verifies if matrices can be multiplied and if the result matrix has correct dimensions
 *
 * Checks that matrix A's columns equal matrix B's rows (required for multiplication),
 * and that the result matrix has dimensions [A.rows × B.columns]
 *
 * @param Result The matrix to store the multiplication result
 * @param A The first input matrix (left operand)
 * @param B The second input matrix (right operand)
 * @return true if dimensions are compatible, false otherwise (with error message to stderr)
 */
bool Matrix_Can_Multiply(const Matrix& Result, const Matrix& A, const Matrix& B);

/**
 * @brief Verifies if matrices can be added/subtracted and if the result matrix has correct dimensions
 *
 * Checks that matrices A and B have identical dimensions,
 * and that the result matrix also has the same dimensions
 *
 * @param Result The matrix to store the addition/subtraction result
 * @param A The first input matrix
 * @param B The second input matrix
 * @return true if dimensions are compatible, false otherwise (with error message to stderr)
 */
bool Matrix_Can_AddOrSubtract(const Matrix& Result, const Matrix& A, const Matrix& B);

/**
 * @brief Verifies if Hadamard product (element-wise multiplication) can be performed
 *
 * Checks that matrices A and B have identical dimensions,
 * and that the result matrix also has the same dimensions
 *
 * @param Result The matrix to store the Hadamard product result
 * @param A The first input matrix
 * @param B The second input matrix
 * @return true if dimensions are compatible, false otherwise (with error message to stderr)
 */
bool Matrix_Can_HadamardProduct(const Matrix& Result, const Matrix& A, const Matrix& B);

/**
 * @brief Verifies if a matrix can be deep-copied into another matrix
 *
 * Checks that the destination and source matrices have identical dimensions
 * and provides detailed output about the dimensions
 *
 * @param copier The destination matrix
 * @param copied The source matrix
 * @return true if dimensions match, false otherwise (with error message to stderr)
 */
bool Matrix_Can_Deep_Copy(const Matrix& copier, const Matrix& copied);

#endif //NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H