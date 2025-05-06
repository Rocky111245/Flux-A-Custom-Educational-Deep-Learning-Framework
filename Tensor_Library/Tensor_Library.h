
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

class Matrix;

/* ==============================  Tensor Class  ============================== */
class Tensor

{
public:
/* ----------------------------- Constructors ----------------------------- */
    Tensor() noexcept;
    Tensor(int rows, int columns, int depth);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

/* ----------------------------- Assignment ----------------------------- */
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

/* ----------------------------- Equality ----------------------------- */
    bool operator==(const Tensor& other) const;

/* ---------------------------------------------------------------------------------
   ------------------------------ GETTER FUNCTIONS SECTION --------------------------
   --------------------------------------------------------------------------------- */
    const float& operator()(int row, int column, int depth) const noexcept;
    void Get_Channel_Matrix(Matrix& destination, int channel_number) const;

    int rows()    const noexcept ;
    int columns() const noexcept ;
    int depth()   const noexcept ;

/* ---------------------------------------------------------------------------------
   ------------------------------ SETTER FUNCTIONS SECTION --------------------------
   --------------------------------------------------------------------------------- */
    float& operator()(int row, int column, int depth) noexcept;
    void   Set_Channel_Matrix(const Matrix& source, int channel_number);

/* ----------------------------- Initialisation ----------------------------- */

void Tensor_Xavier_Uniform(int out_channels);

private:
    int Index(int row, int column, int depth_idx) const noexcept
    {
        return depth_idx * rows_ * columns_ + row * columns_ + column;
    }

    int rows_    = 0;
    int columns_ = 0;
    int depth_   = 0;
    std::unique_ptr<float[]> data_;
};

/* ==============================  Utility Functions  ============================== */
void Tensor_Add_Tensor_ElementWise      (Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Add_Scalar_ElementWise      (Tensor& result, const Tensor& first, float scalar);
void Tensor_Subtract_Tensor_ElementWise (Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Subtract_Scalar_ElementWise (Tensor& result, const Tensor& first, float scalar);
void Tensor_Add_All_Channels            (Matrix& destination, const Tensor& source);
void Tensor_Multiply_Tensor_ElementWise (Tensor& result, const Tensor& first, const Tensor& second);
void Tensor_Multiply_Scalar_ElementWise (Tensor& result, const Tensor& first, float scalar);
void Tensor_Divide_Tensor_ElementWise(Tensor& result,const Tensor& first,const Tensor& second);
void Tensor_Divide_Scalar_ElementWise(Tensor& result,const Tensor& first,float scalar);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_LIBRARY_H
