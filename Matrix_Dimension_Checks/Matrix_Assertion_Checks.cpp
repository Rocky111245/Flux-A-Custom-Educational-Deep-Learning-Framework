// Matrix_Assertion_Checks.cpp
// Created by rakib on 25/3/2025.

#include "Matrix_Dimension_Checks/Matrix_Assertion_Checks.h"


// Check if matrices can be multiplied and if result matrix has correct dimensions
bool Matrix_Can_Multiply(const Matrix& Result, const Matrix& A, const Matrix& B) {
    if (A.columns() != B.rows()) {
        std::cerr << "Matrix multiplication error: dimensions incompatible ["
                  << A.rows() << "x" << A.columns() << "] * ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    if (Result.rows() != A.rows() || Result.columns() != B.columns()) {
        std::cerr << "Matrix multiplication error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

// Check if matrices can be added/subtracted and if result matrix has correct dimensions
bool Matrix_Can_AddOrSubtract(const Matrix& Result, const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.columns() != B.columns()) {
        std::cerr << "Matrix addition/subtraction error: dimensions don't match ["
                  << A.rows() << "x" << A.columns() << "] and ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    if (Result.rows() != A.rows() || Result.columns() != A.columns()) {
        std::cerr << "Matrix addition/subtraction error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << A.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

// Check if Hadamard product can be performed and if result matrix has correct dimensions
bool Matrix_Can_HadamardProduct(const Matrix& Result, const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.columns() != B.columns()) {
        std::cerr << "Hadamard product error: dimensions don't match ["
                  << A.rows() << "x" << A.columns() << "] and ["
                  << B.rows() << "x" << B.columns() << "]" << std::endl;
        return false;
    }

    if (Result.rows() != A.rows() || Result.columns() != A.columns()) {
        std::cerr << "Hadamard product error: result matrix has incorrect dimensions ["
                  << Result.rows() << "x" << Result.columns() << "], expected ["
                  << A.rows() << "x" << A.columns() << "]" << std::endl;
        return false;
    }

    return true;
}

bool Matrix_Can_Deep_Copy(const Matrix& copier, const Matrix& copied) {
    std::cout << "[Deep Copy Eligibility Check]" << std::endl;
    std::cout << "Copier Matrix: [" << copier.rows() << " x " << copier.columns() << "]" << std::endl;
    std::cout << "Copied Matrix: [" << copied.rows() << " x " << copied.columns() << "]" << std::endl;

    if (copier.rows() != copied.rows() || copier.columns() != copied.columns()) {
        std::cerr << "❌ Dimensions do NOT match. Deep copy not safe." << std::endl;
        return false;
    }

    std::cout << "✅ Dimensions match. Safe to perform deep copy." << std::endl;
    return true;
}
