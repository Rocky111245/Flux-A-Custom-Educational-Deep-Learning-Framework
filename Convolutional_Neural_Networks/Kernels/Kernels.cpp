
#include "Kernels.h"
#include "MatrixLibrary.h"



// Returns a tensor with (1 row, size*size columns, all channels in the original kernel)
// Used for im2rows calculations.
Tensor Kernel::Flatten_Kernel() const {
    // Create a flattened tensor with 1 row and size*size columns for each channel
    Tensor flattened_tensor(1, size() * size(), depth());

    // For each channel
    for(int d = 0; d < depth(); d++) {
        int flattened_tensor_index_number = 0;

        // Flatten the 2D kernel into a 1D row
        for(int r = 0; r < size(); r++) {
            for(int c = 0; c < size(); c++) {
                flattened_tensor(0, flattened_tensor_index_number++, d) = (*this)(r, c, d);
            }
        }
    }

    return flattened_tensor;
}

void Kernel::Kernel_Xavier_Uniform(int number_of_kernels)
{
    int fan_in  = depth() * size() * size();
    int fan_out = number_of_kernels * rows() * columns();

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int d = 0; d < depth(); ++d)
        for (int r = 0; r < rows(); ++r)
            for (int c = 0; c < columns(); ++c)
                (*this)(r, c, d) = dist(gen);
}
