
#ifndef NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H
#define NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H

#include <iostream>
#include <MatrixLibrary.h>

// Function declarations for matrix assertion checks
bool Matrix_Can_Multiply(const Matrix& Result, const Matrix& A, const Matrix& B);
bool Matrix_Can_AddOrSubtract(const Matrix& Result, const Matrix& A, const Matrix& B);
bool Matrix_Can_HadamardProduct(const Matrix& Result, const Matrix& A, const Matrix& B);
bool Matrix_Can_Deep_Copy(const Matrix& copier, const Matrix& copied);


#endif //NEWFRAMEWORK_MATRIX_ASSERTION_CHECKS_H

