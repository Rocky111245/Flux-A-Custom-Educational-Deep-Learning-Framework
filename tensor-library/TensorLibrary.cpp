#include <variant>
#include "TensorLibrary.h"
#include "matrix-library/MatrixLibrary.h"


/* ==============================  Constructors  ============================== */

Tensor::Tensor() noexcept = default;

Tensor::Tensor(int rows, int columns, int depth)
        : rows_(rows), columns_(columns), depth_(depth), data_(std::make_unique<float[]>(rows * columns * depth)) {
    if (rows <= 0 || columns <= 0 || depth <= 0)
        throw std::invalid_argument("Tensor dimensions must be positive in main Tensor constructor.");
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

Tensor& Tensor::operator+=(const Tensor& other) {
    if(this->rows() != other.rows() || this->columns() != other.columns() || this->depth() != other.depth()) {
        throw std::invalid_argument("Tensors must have exactly same shape");
    }

    int depth = this->depth();
    int row = this->rows();
    int column = this->columns();
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < column; c++) {
                (*this)(r, c, d) += other(r, c, d);
            }
        }
    }

    return *this;
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
//A special function to directly access binary-serializer of a tensor

const float* Tensor::Get_Data() const noexcept {
    return data_.get();
}

// For non-const tensors - returns read-write access
float* Tensor::Get_Data() noexcept {
    return data_.get();
}

void Tensor::Set_Data(std::unique_ptr<float[]> new_data, size_t expected_size){
    size_t current_size = static_cast<size_t>(rows_) * columns_ * depth_;
    if (expected_size != current_size) {
        throw std::invalid_argument("Data size mismatch in Tensor::Set_Data");
    }
    data_ = std::move(new_data);
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

std::vector<float> Tensor::Get_Row_Vector(int row_number, int channel_number) const {
    if (row_number < 0 || row_number >= rows_)
        throw std::out_of_range("Row number");
    if (channel_number < 0 || channel_number >= depth_)
        throw std::out_of_range("Channel number");
    std::vector<float> destination(columns_, 0.0f);
    for (int c = 0; c < columns_; ++c)
        destination[c] = (*this)(row_number, c, channel_number);
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

void Tensor::Tensor_Xavier_Uniform_Conv(int number_of_kernels) {
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

// For MLP weight tensors: [input_features, output_neurons, batch_or_1]
void Tensor::Tensor_Xavier_Uniform_MLP(int fan_in, int fan_out) {
    if (depth_ != 1) {
        throw std::logic_error("Tensor_Xavier_Uniform_MLP: MLP weights should have depth == 1 for shared parameters.");
    }

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < columns_; ++c)
            (*this)(r, c, 0) = dist(gen);
}

void Tensor::Tensor_Xavier_Uniform_Share_Across_Depth() {
    if (rows_ <= 0 || columns_ <= 0 || depth_ <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive for Xavier initialization.");
    }

    // Calculate fan_in and fan_out for standard Xavier Glorot initialization
    int fan_in = rows_;
    int fan_out = columns_;

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Generate one matrix of values
    std::vector<float> shared_weights(rows_ * columns_);
    for (auto& w : shared_weights) {
        w = dist(gen);
    }

    // Copy same matrix to all depth slices
    for (int d = 0; d < depth_; ++d) {
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < columns_; ++c) {
                (*this)(r, c, d) = shared_weights[r * columns_ + c];
            }
        }
    }
}


void Tensor::Fill(float value) const {
    std::fill_n(data_.get(), (rows_ * columns_ * depth_), value);
}


void Tensor_Broadcast_At_Depth(Tensor& result, const Tensor& input, int target_depth) {
    if (input.depth() != 1) {
        throw std::invalid_argument("Input tensor must have depth = 1 for broadcasting.");
    }

    if (target_depth <= 0) {
        throw std::invalid_argument("Target depth must be a positive integer.");
    }

    int rows = input.rows();
    int cols = input.columns();

    // Create result tensor with target depth
    result = Tensor(rows, cols, target_depth);

    // Copy input slice into every depth slice of result
    for (int d = 0; d < target_depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                result(r, c, d) = input(r, c, 0);
            }
        }
    }
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

//  For tensors of same sizes and only concat the columns (Add dimensions).
void Make_Tensor(Tensor& destination, const std::vector<Tensor>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor vector");
    }

    int rows = inputs[0].rows();
    int cols = inputs[0].columns();
    int depth = inputs[0].depth();


    int num_tensors = inputs.size();

    // Create destination tensor with correct dimensions first
    destination = Tensor(rows, cols * num_tensors, depth);

    // Copy binary-serializer properly respecting the tensor's memory layout
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {
                for (int c = 0; c < cols; ++c) {
                    int dest_col = tensor_idx * cols + c;
                    destination(r, dest_col, d) = inputs[tensor_idx](r, c, d);
                }
            }
        }
    }
}





//This is an example of one of the most efficient ways of doing this operation.However, I have used simpler versions in my library
//so that syntax semantics is not an issue to understanding.
void Tensor_Multiply_Scalar_ElementWise_Fast(Tensor& result, const Tensor& first, float scalar) {
    size_t total_elements = first.rows() * first.columns() * first.depth();
    size_t total_bytes = total_elements * sizeof(float);

    // Step 1: Copy ALL binary-serializer from first to result using memcpy
    std::memcpy(result.Get_Data(), first.Get_Data(), total_bytes);

    // Step 2: Multiply the copied binary-serializer in-place (no more indexing overhead)
    float* result_data = result.Get_Data();
    for (size_t i = 0; i < total_elements; ++i) {
        result_data[i] *= scalar;  // Modify in place
    }
}
//even faster.
void Tensor_Multiply_Scalar_ElementWise_Fastest(Tensor& result, const Tensor& first, float scalar) {
    size_t total_elements = static_cast<size_t>(first.rows()) *
                            first.columns() * first.depth();

    const float* __restrict src = first.Get_Data();
    float* __restrict dst = result.Get_Data();

    for (size_t i = 0; i < total_elements; ++i) {
        dst[i] = src[i] * scalar;
    }
}

//// Arithmetic operations //TODO
// __m256 _mm256_add_ps(__m256 a, __m256 b);      // Addition: a + b
// __m256 _mm256_sub_ps(__m256 a, __m256 b);      // Subtraction: a - b
// __m256 _mm256_mul_ps(__m256 a, __m256 b);      // Multiplication: a * b
// __m256 _mm256_div_ps(__m256 a, __m256 b);      // Division: a / b
//
// // The super powerful one:
// __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);  // a * b + c (all in one instruction!)
//
// // Memory operations
// __m256 _mm256_loadu_ps(const float* mem);      // Load 8 floats from memory
// void _mm256_storeu_ps(float* mem, __m256 a);   // Store 8 floats to memory
//
// // Utility operations
// __m256 _mm256_broadcast_ss(const float* mem);   // Copy one float to all 8 positions
// __m256 _mm256_setzero_ps();                     // Create vector of zeros






