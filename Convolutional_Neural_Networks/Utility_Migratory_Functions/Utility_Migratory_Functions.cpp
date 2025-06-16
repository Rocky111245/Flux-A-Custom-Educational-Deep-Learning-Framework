
#include "Utility_Migratory_Functions.h"
#include "MatrixLibrary.h"
#include "Convolutional_Neural_Networks/Kernels/Kernels.h"
//Convolution Related Matrix Functions to be migrated to the main library later on




Matrix MNIST_Single_Targeted_Image_To_Matrix(const std::string& filename, int image_index) {
    const int header_size = 16;        // First 16 bytes = metadata
    const int rows = 28;               // MNIST image rows
    const int cols = 28;               // MNIST image columns

    const int image_size = rows * cols; // 784 pixels per image

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open MNIST file: " + filename);
    }

    // Move file pointer to the start of the desired image
    file.seekg(header_size + image_index * image_size, std::ios::beg);

    // Read pixel data
    std::vector<unsigned char> buffer(image_size);
    file.read(reinterpret_cast<char*>(buffer.data()), image_size);

    // Create your matrix
    Matrix mnist_image(rows, cols);

    // Fill matrix with normalized pixel values (0.0 to 1.0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mnist_image(i, j) = static_cast<float>(buffer[i * cols + j]) / 255.0f;
        }
    }

    return mnist_image;
}

void Print_MNIST_Image_In_Console(const Matrix& mnist_image) {
    const int rows = mnist_image.rows();
    const int cols = mnist_image.columns();

    std::cout << "+" << std::string(cols * 2, '-') << "+" << std::endl;

    for (int i = 0; i < rows; ++i) {
        std::cout << "|";
        for (int j = 0; j < cols; ++j) {
            float pixel_value = mnist_image(i, j);

            // Map pixel values to ASCII characters based on intensity
            if (pixel_value > 0.8f) std::cout << "#";
            else if (pixel_value > 0.6f) std::cout << "&";
            else if (pixel_value > 0.4f) std::cout <<"*";
            else if (pixel_value > 0.2f) std::cout << "*";
            else std::cout << "  ";
        }
        std::cout << "|" << std::endl;
    }

    std::cout << "+" << std::string(cols * 2, '-') << "+" << std::endl;
}




void Extract_Patches_Im2rows(Tensor &im2rows_tensor,const Tensor &padded_input_tensor, const int kernel_size,const int stride_size) {
    int output_rows = (padded_input_tensor.rows() - kernel_size) / stride_size + 1;
    int output_columns = (padded_input_tensor.columns() - kernel_size) / stride_size + 1;

    int padded_input_rows = padded_input_tensor.rows();
    int padded_input_columns = padded_input_tensor.columns();

    // Initializing im2rows protocol for dot product calculation
    int im2rows_rows = output_columns * output_rows;
    int im2rows_columns = kernel_size * kernel_size;

    // Resizing the im2rows tensor in row major order
    im2rows_tensor = Tensor(im2rows_rows, im2rows_columns, padded_input_tensor.depth());

    // For all channels
    for(int depth_number = 0; depth_number < padded_input_tensor.depth(); depth_number++) {
        int row_number = 0; // Reset row_number for each channel

        // Process each vertical stride position
        for(int global_vertical_index = 0; global_vertical_index + kernel_size <= padded_input_rows; global_vertical_index += stride_size) {
            // Process each horizontal stride position
            for(int global_horizontal_index = 0; global_horizontal_index + kernel_size <= padded_input_columns; global_horizontal_index += stride_size) {

                int column_number = 0; // Reset column_number for each patch

                // Extract the patch kernel_size × kernel_size
                for(int i = global_vertical_index; i < global_vertical_index + kernel_size; i++) {
                    for(int j = global_horizontal_index; j < global_horizontal_index + kernel_size; j++) {
                        // Store the value in the im2rows tensor
                        im2rows_tensor(row_number, column_number, depth_number) = padded_input_tensor(i, j, depth_number);
                        column_number++;
                    }
                }

                // Move to the next row in im2rows (next patch)
                row_number++;
            }
        }
    }
}

//this automatically calculates the padding required for the desired kernel size and the output
int Padding_Size_Needed(int input_rows, int input_columns,int desired_output_rows,int desired_output_columns, int kernel_size,int stride) {
    int padding_size=(stride*(desired_output_rows-1)+kernel_size-input_rows)/2;
}

//padded matrix is the final result of the input padding
//input matrix is the returned input matrix from the MNIST_Single_Targeted_Image_To_Matrix function or any function which extracts input
void Tensor_Padding_Symmetric(Tensor &padded_tensor, const Tensor &input_tensor, int padding_size) {
    int depth=input_tensor.depth();
    int row=input_tensor.rows();
    int column=input_tensor.columns();

    // Create a new tensor with dimensions increased by padding_size*2 (padding on all sides). It also automatically initializes the values to 0.
    padded_tensor = Tensor(row + 2 * padding_size, column + 2 * padding_size, depth);

    // Copy the values from the input matrix to the center of the padded matrix
    for (int d=0;d<depth;d++){
        for (int r=0;r<row;r++){
            for(int c=0;c<column;c++){
                padded_tensor(r + padding_size, c + padding_size, d)=input_tensor(r,c,d);
            }
        }
    }
}


