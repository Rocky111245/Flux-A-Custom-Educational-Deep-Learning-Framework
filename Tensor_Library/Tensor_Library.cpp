#include <variant>
#include "Tensor_Library.h"
#include "MatrixLibrary.h"

/* ==============================  Constructors  ============================== */

Tensor::Tensor() noexcept = default;

Tensor::Tensor(int rows, int columns, int depth)
        : rows_(rows), columns_(columns), depth_(depth), data_(std::make_unique<float[]>(rows * columns * depth)) {
    if (rows <= 0 || columns <= 0 || depth <= 0)
        throw std::invalid_argument("Tensor dimensions must be positive.");
    std::fill(data_.get(), data_.get() + rows * columns * depth, 0.0f);
}

Tensor::Tensor(const Tensor& other)  // Copy constructor
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
          data_(std::make_unique<float[]>(rows_ * columns_ * depth_)) {
    std::copy(other.data_.get(),
              other.data_.get() + rows_ * columns_ * depth_,
              data_.get());
}

Tensor::Tensor(Tensor&& other) noexcept  // Move constructor
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
          data_(std::move(other.data_)) {
    other.rows_ = other.columns_ = other.depth_ = 0;
}

/* ==============================  Assignment Operators  ============================== */

Tensor& Tensor::operator=(const Tensor& other) {  // Copy assignment
    if (this != &other) {
        if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_) {
            rows_ = other.rows_;
            columns_ = other.columns_;
            depth_ = other.depth_;
            data_ = std::make_unique<float[]>(rows_ * columns_ * depth_);
        }
        std::copy(other.data_.get(),
                  other.data_.get() + rows_ * columns_ * depth_,
                  data_.get());
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {  // Move assignment
    if (this != &other) {
        rows_ = other.rows_;
        columns_ = other.columns_;
        depth_ = other.depth_;
        data_ = std::move(other.data_);
        other.rows_ = other.columns_ = other.depth_ = 0;
    }
    return *this;
}

/* ==============================  Comparison Operators  ============================== */

bool Tensor::operator==(const Tensor& other) const {
    constexpr float epsilon = 1e-5f;

    if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_)
        return false;

    int total_elements = rows_ * columns_ * depth_;
    for (int i = 0; i < total_elements; ++i) {
        if (std::fabs(data_[i] - other.data_[i]) > epsilon)
            return false;
    }
    return true;
}

/* ==============================  Element Access Operators  ============================== */

const float& Tensor::operator()(int row, int column, int depth) const noexcept {  // Read access
    return data_[Index(row, column, depth)];
}

float& Tensor::operator()(int row, int column, int depth) noexcept {  // Write access
    return data_[Index(row, column, depth)];
}

/* ==============================  Dimension Getters  ============================== */

int Tensor::rows() const noexcept {
    return rows_;
}

int Tensor::columns() const noexcept {
    return columns_;
}

int Tensor::depth() const noexcept {
    return depth_;
}

/* ==============================  Memory Allocation Functions  ============================== */

std::variant<Matrix, Tensor> Memory_Allocation(Single_Tensor_Dependent_Operations unary_operation_types, const Tensor& input) {
    switch (unary_operation_types) {
        case Single_Tensor_Dependent_Operations::Tensor_Add_Tensor_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Add_Scalar_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Subtract_Tensor_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Subtract_Scalar_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Add_All_Channels: {
            Matrix result(input.rows(), input.columns(), 0.0f);
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Multiply_Tensor_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Multiply_Scalar_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Divide_Tensor_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Divide_Scalar_ElementWise: {
            Tensor result(input.rows(), input.columns(), input.depth());
            return result;
        }
        case Single_Tensor_Dependent_Operations::Tensor_Transpose: {
            Tensor result(input.columns(), input.rows(), input.depth());
            return result;
        }

    }
    throw std::invalid_argument("Unknown operation type in Memory_Allocation");
}

std::variant<Matrix, Tensor> Memory_Allocation(Multi_Tensor_Dependent_Operations binary_operation_types, const Tensor& first_input, const Tensor& second_input) {
    switch (binary_operation_types) {
        case Multi_Tensor_Dependent_Operations::Tensor_Multiply_Tensor: {
            if (first_input.depth() != second_input.depth())
                throw std::invalid_argument("Depth mismatch in Tensor_Multiply_Tensor");
            if (first_input.columns() != second_input.rows()) {
                throw std::invalid_argument("Number of columns in the first tensor must equal the number of rows in the second tensor.");
            }
            Tensor result(first_input.rows(), second_input.columns(), first_input.depth());
            return result;
        }
    }
    throw std::invalid_argument("Unknown binary operation type in Memory_Allocation");
}

/* ==============================  Channel Operations  ============================== */

Matrix Tensor::Get_Channel_Matrix(int channel_number) const {
    Matrix destination = Matrix(rows_, columns_, 0.0f);
    if (channel_number < 0 || channel_number >= depth_)
        throw std::out_of_range("Channel number");
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < columns_; ++c)
            destination(r, c) = (*this)(r, c, channel_number);
    return destination;
}

void Tensor::Set_Channel_Matrix(const Matrix& source, int channel_number) {
    if (channel_number < 0 || channel_number >= depth_)
        throw std::out_of_range("Channel number");
    if (source.rows() != rows_ || source.columns() != columns_)
        throw std::invalid_argument("Matrix dimensions must match tensor slice");
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < columns_; ++c)
            (*this)(r, c, channel_number) = source(r, c);
}

/* ==============================  In-Place Operations  ============================== */

void Tensor::Multiply_ElementWise_Inplace(const Tensor& other) {
    if (this->rows() != other.rows() || this->columns() != other.columns() || this->depth() != other.depth()) {
        throw std::invalid_argument("Tensor dimensions do not match the dimensions required for element-wise multiplication.");
    }
    for (int d = 0; d < depth_; ++d)
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < columns_; ++c)
                (*this)(r, c, d) = (*this)(r, c, d) * other(r, c, d);
}

/* ==============================  Initialization Functions  ============================== */

void Tensor::Tensor_Xavier_Uniform(int number_of_kernels) {
    int fan_in = depth_ * rows_ * columns_;
    int fan_out = number_of_kernels * rows_ * columns_;

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int d = 0; d < depth_; ++d)
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < columns_; ++c)
                (*this)(r, c, d) = dist(gen);
}

/* ==============================  Standalone Utility Functions  ============================== */

/* ----------------------------- Element-wise Tensor Operations ----------------------------- */

void Tensor_Add_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    if (first.rows() != second.rows() || first.columns() != second.columns() || first.depth() != second.depth()) {
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                result(r, c, d) = first(r, c, d) + second(r, c, d);
            }
        }
    }
}

void Tensor_Subtract_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    if (first.rows() != second.rows() || first.columns() != second.columns() || first.depth() != second.depth()) {
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                result(r, c, d) = first(r, c, d) - second(r, c, d);
            }
        }
    }
}

void Tensor_Multiply_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    if (first.rows() != second.rows() || first.columns() != second.columns() || first.depth() != second.depth()) {
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < row; ++r)
            for (int c = 0; c < column; ++c)
                result(r, c, d) = first(r, c, d) * second(r, c, d);
}

void Tensor_Divide_Tensor_ElementWise(Tensor& result, const Tensor& first, const Tensor& second) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < row; ++r)
            for (int c = 0; c < column; ++c)
                result(r, c, d) = first(r, c, d) / second(r, c, d);
}

/* ----------------------------- Scalar Operations ----------------------------- */

void Tensor_Add_Scalar_ElementWise(Tensor& result, const Tensor& first, const float scalar) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                result(r, c, d) = first(r, c, d) + scalar;
            }
        }
    }
}

void Tensor_Subtract_Scalar_ElementWise(Tensor& result, const Tensor& first, const float scalar) {
    int depth = first.depth();
    int row = first.rows();
    int column = first.columns();
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                result(r, c, d) = first(r, c, d) - scalar;
            }
        }
    }
}

void Tensor_Multiply_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar) {
    int depth = first.depth();
    int rows = first.rows();
    int columns = first.columns();
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < columns; ++c)
                result(r, c, d) = first(r, c, d) * scalar;
}

void Tensor_Divide_Scalar_ElementWise(Tensor& result, const Tensor& first, float scalar) {
    int depth = first.depth();
    int rows = first.rows();
    int columns = first.columns();
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < columns; ++c)
                result(r, c, d) = first(r, c, d) / scalar;
}

void Tensor_Transpose(Tensor& result, const Tensor& input) {
    // Transpose the first two dimensions while keeping the third (batch) dimension intact
    // Input: [rows, columns, depth] -> Output: [columns, rows, depth]

    if (result.rows() != input.columns() ||
        result.columns() != input.rows() ||
        result.depth() != input.depth()) {
        throw std::invalid_argument("Result tensor dimensions must match transposed input dimensions");
    }

    // Transpose each batch channel independently
    for (int batch_idx = 0; batch_idx < input.depth(); ++batch_idx) {
        for (int row = 0; row < input.rows(); ++row) {
            for (int col = 0; col < input.columns(); ++col) {
                // Swap row and column indices for transpose
                result(col, row, batch_idx) = input(row, col, batch_idx);
            }
        }
    }
}

/* ----------------------------- Tensor-Matrix Operations ----------------------------- */

void Tensor_Add_All_Channels(Matrix& destination, const Tensor& source) {
    if (destination.rows() != source.rows() || destination.columns() != source.columns())
        throw std::invalid_argument("Matrix dimensions must match");
    for (int d = 0; d < source.depth(); ++d)
        for (int r = 0; r < source.rows(); ++r)
            for (int c = 0; c < source.columns(); ++c)
                destination(r, c) += source(r, c, d);
}

/* ----------------------------- Tensor Multiplication ----------------------------- */

void Tensor_Multiply_Tensor(Tensor& result, const Tensor& first, const Tensor& second) {
    int result_depth = result.depth();

    if (&result == &first || &result == &second) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (first.columns() != second.rows() || first.depth() != second.depth()) {
        throw std::invalid_argument("Number of columns in the first tensor must equal the number of rows in the second tensor. Depth must also match");
    }

    if (result.rows() != first.rows() || result.columns() != second.columns() || result.depth() != first.depth()) {
        throw std::invalid_argument("Result tensor dimensions do not match the dimensions required for multiplication.");
    }

    for (int d = 0; d < result_depth; ++d) {
        for (int i = 0; i < first.rows(); ++i) {
            for (int j = 0; j < second.columns(); ++j) {
                float sum = 0.0f;
                for (int k = 0; k < first.columns(); ++k) {
                    sum += first(i, k, d) * second(k, j, d);
                }
                result(i, j, d) = sum;
            }
        }
    }
}