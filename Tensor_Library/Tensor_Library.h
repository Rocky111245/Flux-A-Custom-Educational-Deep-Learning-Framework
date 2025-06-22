#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_LIBRARY_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_LIBRARY_H

/**
 *  Tensor – minimal 3‑D float container.
 *  Memory layout: depth (slow) → row → column (fast)
 */

#include <cstddef>      // std::size_t
#include <memory>       // std::unique_ptr
#include <random>       // Xavier initializer
#include <stdexcept>    // std::invalid_argument
#include <algorithm>    // std::fill, std::copy
#include <cassert>      // optional bounds checking
#include <variant>      // std::variant

// Forward declaration
class Matrix;

/* ==============================  Operation Type Enums  ============================== */
enum class Single_Tensor_Dependent_Operations {
    Tensor_Add_Tensor_ElementWise,
    Tensor_Add_Scalar_ElementWise,
    Tensor_Subtract_Tensor_ElementWise,
    Tensor_Subtract_Scalar_ElementWise,
    Tensor_Add_All_Channels,
    Tensor_Multiply_Tensor_ElementWise,
    Tensor_Multiply_Scalar_ElementWise,
    Tensor_Divide_Tensor_ElementWise,
    Tensor_Divide_Scalar_ElementWise,
    Tensor_Transpose
};

enum class Multi_Tensor_Dependent_Operations {
    Tensor_Multiply_Tensor
};

/* ==============================  Tensor Class  ============================== */
class Tensor {
public:
    /* ----------------------------- Constructors ----------------------------- */
    Tensor() noexcept;
    Tensor(int rows, int columns, int depth);
    Tensor(const Tensor& other);                    // Copy constructor
    Tensor(Tensor&& other) noexcept;               // Move constructor

    /* ----------------------------- Assignment Operators ----------------------------- */
    Tensor& operator=(const Tensor& other);        // Copy assignment
    Tensor& operator=(Tensor&& other) noexcept;   // Move assignment

    /* ----------------------------- Comparison Operators ----------------------------- */
    bool operator==(const Tensor& other) const;

    /* ----------------------------- Element Access Operators ----------------------------- */
    const float& operator()(int row, int column, int depth) const noexcept;  // Read access
    float& operator()(int row, int column, int depth) noexcept;              // Write access

    /* ----------------------------- Dimension Getters ----------------------------- */
    int rows() const noexcept;
    int columns() const noexcept;
    int depth() const noexcept;



    /* ----------------------------- Channel Operations ----------------------------- */
    Matrix Get_Channel_Matrix(int channel_number) const;
    void Set_Channel_Matrix(const Matrix& source, int channel_number);

    /* ----------------------------- In-Place Operations ----------------------------- */
    void Multiply_ElementWise_Inplace(const Tensor& other);

    /* ----------------------------- Initialization Functions ----------------------------- */
    void Tensor_Xavier_Uniform(int number_of_kernels);

private:
    /* ----------------------------- Helper Functions ----------------------------- */
    int Index(int row, int column, int depth_idx) const noexcept {
        return depth_idx * rows_ * columns_ + row * columns_ + column;
    }

    /* ----------------------------- Member Variables ----------------------------- */
    int rows_ = 0;
    int columns_ = 0;
    int depth_ = 0;
    std::unique_ptr<float[]> data_;
};

/* ==============================  Standalone Utility Functions  ============================== */

/* ----------------------------- Element-wise Tensor Operations ----------------------------- */
void Tensor_Add_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Subtract_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Multiply_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Divide_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second);

/* ----------------------------- Scalar Operations ----------------------------- */
void Tensor_Add_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Subtract_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Multiply_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Divide_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);

/* ----------------------------- Tensor-Matrix Operations ----------------------------- */
void Tensor_Add_All_Channels(Matrix& destination, const Tensor& source);
void Tensor_Transpose(Tensor& result, const Tensor& input);


/* ----------------------------- Tensor Multiplication ----------------------------- */
void Tensor_Multiply_Tensor(Tensor& result, const Tensor& first, const Tensor& second);

/* ----------------------------- Memory Allocation Functions ----------------------------- */
std::variant<Matrix, Tensor> Memory_Allocation(Single_Tensor_Dependent_Operations operation_types, const Tensor& input);
std::variant<Matrix, Tensor> Memory_Allocation(Multi_Tensor_Dependent_Operations binary_operation_types, const Tensor& first_input, const Tensor& second_input);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_LIBRARY_H