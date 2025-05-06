#include "Tensor_Library.h"
#include "MatrixLibrary.h"

/* ----------------------------- Constructors ----------------------------- */

Tensor::Tensor() noexcept = default;

Tensor::Tensor(int rows, int columns, int depth)
        : rows_(rows), columns_(columns), depth_(depth),data_(std::make_unique<float[]>(rows * columns * depth))
{
    if (rows<=0 || columns<=0 || depth<=0)
        throw std::invalid_argument("Tensor dimensions must be positive.");
    std::fill(data_.get(), data_.get() + rows*columns*depth, 0.0f);
}

/* ----------------------------Copy Constructor --------------------------*/
Tensor::Tensor(const Tensor& other)
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
        data_(std::make_unique<float[]>(rows_ * columns_ * depth_))
{
    std::copy(other.data_.get(),
              other.data_.get() + rows_ * columns_ * depth_,
              data_.get());
}


/* ----------------------------Move Constructor --------------------------*/
Tensor::Tensor(Tensor&& other) noexcept
        : rows_(other.rows_), columns_(other.columns_), depth_(other.depth_),
          data_(std::move(other.data_))
{
    other.rows_ = other.columns_ = other.depth_ = 0;
}


/* --------------------------Copy Assignment Operator --------------------------*/
Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other)
    {
        if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_)
        {
            rows_    = other.rows_;
            columns_ = other.columns_;
            depth_   = other.depth_;
            data_    = std::make_unique<float[]>(rows_ * columns_ * depth_);
        }
        std::copy(other.data_.get(),
                  other.data_.get() + rows_ * columns_ * depth_,
                  data_.get());
    }
    return *this;
}


/* --------------------------Move Assignment Operator ----------------------*/
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        rows_    = other.rows_;
        columns_ = other.columns_;
        depth_   = other.depth_;
        data_    = std::move(other.data_);
        other.rows_ = other.columns_ = other.depth_ = 0;
    }
    return *this;
}

/* --------------------------Equality Check Operator ----------------------*/

bool Tensor::operator==(const Tensor& other) const
{
    constexpr float epsilon = 1e-5f;  // Acceptable difference

    if (rows_ != other.rows_ || columns_ != other.columns_ || depth_ != other.depth_)
        return false;

    int total_elements = rows_ * columns_ * depth_;
    for (int i = 0; i < total_elements; ++i)
    {
        if (std::fabs(data_[i] - other.data_[i]) > epsilon)
            return false;
    }

    return true;
}

/* ---------------------------------------------------------------------------------
  ------------------------------GETTER FUNCTIONS SECTION----------------------------
  --------------------------------------------------------------------------------*/


/* ----------------------------- Element Access --------------------------- */

int Tensor::rows() const noexcept{
    return rows_;
}
int Tensor::columns() const noexcept
{ return columns_; }
int Tensor::depth()   const noexcept
{ return depth_;   }


const float& Tensor::operator()(int row, int column, int depth) const noexcept
{
    return data_[ Index(row, column, depth) ];
}


/* ----------------------------- Get Channel Copy ----------------------------- */

void Tensor::Get_Channel_Matrix(Matrix& destination, int channel_number) const
{
    if (channel_number < 0 || channel_number >= depth_) throw std::out_of_range("Channel number");
    destination = Matrix(rows_, columns_, 0.0f);
    for (int r=0; r<rows_; ++r)
        for (int c=0; c<columns_; ++c)
            destination(r, c) = (*this)(r, c, channel_number);
}


/* ---------------------------------------------------------------------------------
  ------------------------------SETTER FUNCTIONS SECTION----------------------------
  --------------------------------------------------------------------------------*/

/* ----------------------------- Set Datapoint ----------------------------- */


float& Tensor::operator()(int row, int column, int depth) noexcept
{
    return data_[ Index(row, column, depth) ];
}

/* ----------------------------- Set Channel (replaces whole channels using pre-constructed matrices) ----------------------------- */
void Tensor::Set_Channel_Matrix(const Matrix& source, int channel_number)
{
    if (channel_number < 0 || channel_number >= depth_)
        throw std::out_of_range("Channel number");

    if (source.rows() != rows_ || source.columns() != columns_)
        throw std::invalid_argument("Matrix dimensions must match tensor slice");

    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < columns_; ++c)
            (*this)(r, c, channel_number) = source(r, c);
}


/* ------------------------- Xavier Uniform Init ------------------------- */

// Proper Xavier Uniform Initialization for Conv Kernels->Move to kernel class later
void Tensor::Tensor_Xavier_Uniform(int out_channels)
{
    int fan_in  = depth_ * rows_ * columns_;
    int fan_out = out_channels * rows_ * columns_;

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int d = 0; d < depth_; ++d)
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < columns_; ++c)
                (*this)(r, c, d) = dist(gen);
}



/* ---------------------------------------------------------------------------------
  ------------------------------UTILITY FUNCTIONS SECTION----------------------------
  --------------------------------------------------------------------------------*/

/* --------------------- Tensor Utilities -------------------- */

void Tensor_Add_Tensor_ElementWise(Tensor& result,const Tensor& first,const Tensor& second){
    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    result=Tensor(row,column,depth);

    if(!(first==second)){
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for(int d=0;d<depth;d++){
        for(int r=0;r<row;r++){
            for(int c=0;c<column;c++){
                result(r,c,d)=first(r,c,d)+second(r,c,d);
            }
        }
    }
}


void Tensor_Add_Scalar_ElementWise(Tensor& result,const Tensor& first,const float scalar){
    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    result=Tensor(row,column,depth);

    for(int d=0;d<depth;d++){
        for(int r=0;r<row;r++){
            for(int c=0;c<column;c++){
                result(r,c,d)=first(r,c,d)+scalar;
            }
        }
    }
}

/* -----------------Element‑wise Subtraction (first (Tensor) - second (Tensor)) ----------------- */
void Tensor_Subtract_Tensor_ElementWise(Tensor& result,const Tensor& first,const Tensor& second){
    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    result=Tensor(row,column,depth);

    if(!(first==second)){
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for(int d=0;d<depth;d++){
        for(int r=0;r<row;r++){
            for(int c=0;c<column;c++){
                result(r,c,d)=first(r,c,d)-second(r,c,d);
            }
        }
    }
}

/* -----------------Element‑wise Subtraction (first (Tensor) - scalar (float)) ----------------- */
void Tensor_Subtract_Scalar_ElementWise(Tensor& result,const Tensor& first,const float scalar){
    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    result=Tensor(row,column,depth);

    for(int d=0;d<depth;d++){
        for(int r=0;r<row;r++){
            for(int c=0;c<column;c++){
                result(r,c,d)=first(r,c,d)-scalar;
            }
        }
    }
}

/* -----------------Element‑wise Addition and Compression to 2D Channel (Tensor + Tensor) ----------------- */
void Tensor_Add_All_Channels(Matrix& destination, const Tensor& source)
{
    // Resize destination to match source's 2D dimensions
    destination = Matrix(source.rows(), source.columns(), 0.0f);

    for (int d = 0; d < source.depth(); ++d)
        for (int r = 0; r < source.rows(); ++r)
            for (int c = 0; c < source.columns(); ++c)
                destination(r, c) += source(r, c, d);
}


/* -----------------Element‑wise Multiply (Tensor × Tensor)------------------------------------------------ */
void Tensor_Multiply_Tensor_ElementWise(Tensor& result,const Tensor& first,const Tensor& second)
{

    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    if (!(first == second))  {
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < row; ++r)
            for (int c = 0; c < column; ++c)
                result(r, c, d) = first(r, c, d) * second(r, c, d);
}

/* --------------  Element‑wise Multiply (Tensor × Scalar) ----------------------------------------------- */
void Tensor_Multiply_Scalar_ElementWise(Tensor& result,const Tensor& first,float scalar)
{
    int depth = first.depth();
    int rows = first.rows();
    int columns = first.columns();

    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < columns; ++c)
                result(r, c, d) = first(r, c, d) * scalar;

}


/* -----------------Element‑wise Division (Tensor / Tensor)------------------------------------------------ */
void Tensor_Divide_Tensor_ElementWise(Tensor& result,const Tensor& first,const Tensor& second)
{

    int depth=first.depth();
    int row=first.rows();
    int column=first.columns();

    if (!(first == second))  {
        throw std::invalid_argument("Tensor dimensions must match");
    }
    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < row; ++r)
            for (int c = 0; c < column; ++c)
                result(r, c, d) = first(r, c, d) / second(r, c, d);
}

/* -------------- Element‑wise Division (Tensor × Scalar) ----------------------------------------------- */
void Tensor_Divide_Scalar_ElementWise(Tensor& result,const Tensor& first,float scalar)
{
    int depth = first.depth();
    int rows = first.rows();
    int columns = first.columns();

    for (int d = 0; d < depth; ++d)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < columns; ++c)
                result(r, c, d) = first(r, c, d) / scalar;

}



