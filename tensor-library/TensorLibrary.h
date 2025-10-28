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
#include <cstring>      // For memcpy

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
    ~Tensor() = default;                           // Destructor

    /* ----------------------------- Assignment Operators ----------------------------- */
    Tensor& operator=(const Tensor& other);        // Copy assignment
    Tensor& operator=(Tensor&& other) noexcept;   // Move assignment

    /* ----------------------------- Comparison Operators ----------------------------- */
    bool operator==(const Tensor& other) const;


    Tensor& operator+=(const Tensor& other);
    /* ----------------------------- Element Access Operators ----------------------------- */
    const float& operator()(int row, int column, int depth) const noexcept;  // Read access
    float& operator()(int row, int column, int depth) noexcept;              // Write access

    /* ----------------------------- Dimension Getters ----------------------------- */
    int rows() const noexcept;
    int columns() const noexcept;
    int depth() const noexcept;
    const float* Get_Data() const noexcept;
    float* Get_Data() noexcept;
    std::vector<float> Get_Row_Vector(int row_number, int channel_number) const;
    void Fill(float value) const;
    void Set_Data(std::unique_ptr<float[]> new_data, size_t expected_size);


    std::unique_ptr<float[]> Get_Data(const std::vector<Tensor>& inputs);

    /* ----------------------------- Channel Operations ----------------------------- */
    Matrix Get_Channel_Matrix(int channel_number) const;
    void Set_Channel_Matrix(const Matrix& source, int channel_number);

    /* ----------------------------- In-Place Operations ----------------------------- */
    void Multiply_ElementWise_Inplace(const Tensor& other);

    /* ----------------------------- Initialization Functions ----------------------------- */
    void Tensor_Xavier_Uniform_Conv(int number_of_kernels);
    void Tensor_Xavier_Uniform_MLP(int fan_in, int fan_out);
    void Tensor_Xavier_Uniform_Share_Across_Depth() ;


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
void Tensor_Multiply_Scalar_ElementWise_Fastest(Tensor& result, const Tensor& first, float scalar);

/* ----------------------------- Scalar Operations ----------------------------- */
void Tensor_Add_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Subtract_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Multiply_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);
void Tensor_Divide_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar);

/* ----------------------------- Tensor-Matrix Operations ----------------------------- */
void Tensor_Add_All_Channels(Matrix& destination, const Tensor& source);
void Tensor_Transpose(Tensor& result, const Tensor& input);
void Tensor_Broadcast_At_Depth(Tensor& result, const Tensor& input, int target_depth);

/* ----------------------------- Tensor Multiplication ----------------------------- */
void Tensor_Multiply_Tensor(Tensor& result, const Tensor& first, const Tensor& second);

/* ----------------------------- Memory Allocation Functions ----------------------------- */
std::variant<Matrix, Tensor> Memory_Allocation(Single_Tensor_Dependent_Operations operation_types, const Tensor& input);
std::variant<Matrix, Tensor> Memory_Allocation(Multi_Tensor_Dependent_Operations binary_operation_types, const Tensor& first_input, const Tensor& second_input);


#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_LIBRARY_H