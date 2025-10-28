#ifndef MATRIX_LIBRARY_H
#define MATRIX_LIBRARY_H

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <random>


class Matrix {
public:
    Matrix();  // Default Constructor
    Matrix(int rows, int columns);
    Matrix(int rows, int columns, float value);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;

    //Operator Overloading
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    static void Print(const Matrix& matrix);
    float& operator()(int row, int column);
    const float& operator()(int row, int column) const;

    //Getter Functions
    int rows() const;
    int columns() const;
    const float *Matrix_Get_All_Data() const;
    const float& Matrix_Get_Selected_Data(const int index) const;

    //Setter Functions
    void Matrix_Set_Data(const float value, int iteration_number);

private:
    int rows_;
    int columns_;
    std::unique_ptr<float[]> data_;
};

// Core Operations
void Matrix_Multiply(Matrix& result, const Matrix& first, const Matrix& second);
Matrix Matrix_AutoCreate(const Matrix& first, const Matrix& second);
void Matrix_Add(Matrix& result, const Matrix& matrix1, const Matrix& matrix2);
void Matrix_Subtract(Matrix& result, const Matrix& matrix1, const Matrix& matrix2);
void Matrix_Transpose(Matrix& final, const Matrix& original);
void Matrix_Fill(Matrix& matrix, float value);
// Neural Network Operations
void Matrix_Hadamard_Product(Matrix& result, const Matrix& a, const Matrix& b);
void Matrix_Broadcast(Matrix& result, const Matrix& original, int newRows, int newColumns);
void Matrix_Scalar_Multiply(Matrix& matrix, float scalar);
float Matrix_Sum_All_Elements(const Matrix& matrix);
void Matrix_Power(Matrix& matrix, float power);
void Matrix_Absolute(Matrix& result, const Matrix& original);
void Matrix_Sum_Columns(Matrix& dest, const Matrix& src);

// Randomization Functions
void Matrix_Randomize(Matrix& matrix, float range = 3.0f);
void Matrix_Xavier_Uniform(Matrix& matrix);

// Input Output Preprocessing Functions
Matrix Matrix_Data_Preprocessor(int desiredRows, int desiredColumns, int stride, int step, const std::vector<std::vector<float>>& data);

#endif // MATRIX_LIBRARY_H
