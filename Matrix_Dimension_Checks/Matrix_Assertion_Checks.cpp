/**
 * @file Matrix_Assertion_Checks.cpp
 * @brief Implementation of utility functions for matrix dimension compatibility checks
 * @author rakib
 * @date 25/3/2025
 */

#include "Matrix_Dimension_Checks/Matrix_Assertion_Checks.h"

bool Matrix_Can_Multiply(const Matrix& Result, const Matrix& A, const Matrix& B) {
    // Check if matrices A and B can be multiplied (A.columns must equal B.rows)
    if (A.columns() != B.rows()) {
        std::cerr << "Matrix multiplication error: dimensions incompatible ["
                  << A.rows() << "x" << A.columns() << "] * ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    // Check if result matrix has appropriate dimensions (A.rows × B.columns)
    if (Result.rows() != A.rows() || Result.columns() != B.columns()) {
        std::cerr << "Matrix multiplication error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

bool Matrix_Can_AddOrSubtract(const Matrix& Result, const Matrix& A, const Matrix& B) {
    // Check if matrices A and B have the same dimensions
    if (A.rows() != B.rows() || A.columns() != B.columns()) {
        std::cerr << "Matrix addition/subtraction error: dimensions don't match ["
                  << A.rows() << "x" << A.columns() << "] and ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    // Check if result matrix has the same dimensions as input matrices
    if (Result.rows() != A.rows() || Result.columns() != A.columns()) {
        std::cerr << "Matrix addition/subtraction error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << A.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

bool Matrix_Can_HadamardProduct(const Matrix& Result, const Matrix& A, const Matrix& B) {
    // Check if matrices A and B have the same dimensions (required for element-wise operations)
    if (A.rows() != B.rows() || A.columns() != B.columns()) {
        std::cerr << "Hadamard product error: dimensions don't match ["
                  << A.rows() << "x" << A.columns() << "] and ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    // Check if result matrix has the same dimensions as input matrices
    if (Result.rows() != A.rows() || Result.columns() != A.columns()) {
        std::cerr << "Hadamard product error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << A.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

bool Matrix_Can_Deep_Copy(const Matrix& copier, const Matrix& copied) {
    // Display dimensions for both matrices for debugging purposes
    std::cout << "[Deep Copy Eligibility Check]" << std::endl;
    std::cout << "Copier Matrix: [" << copier.rows() << " x " << copier.columns() << "]" << std::endl;
    std::cout << "Copied Matrix: [" << copied.rows() << " x " << copied.columns() << "]" << std::endl;

    // Check if matrices have the same dimensions
    if (copier.rows() != copied.rows() || copier.columns() != copied.columns()) {
        std::cerr << "❌ Dimensions do NOT match. Deep copy not safe." << std::endl;
        return false;
    }

    std::cout << "✅ Dimensions match. Safe to perform deep copy." << std::endl;
    return true;
}