//
// Created by rakib on 17/5/2025.
//

#include "Tensor_Padding.h"

/*-------------------------- Padding Calculation Functions -------------------------*/
/* Calculate padding needed for height dimension */

int Get_Output_Rows(int input_rows,int kernel_size, int stride_size, PaddingStrategy strategy) {

    switch (strategy) {
        case SAME:
            return  ceil((float)input_rows / stride_size);

        case VALID:
            return  floor((input_rows - kernel_size) / stride_size) + 1;

    }

}

int Get_Output_Columns(int input_columns, int kernel_size, int stride_size, PaddingStrategy strategy){


    switch (strategy) {
        case SAME:
            return  std::ceil((float)input_columns / stride_size);

        case VALID:
            return std::floor((input_columns - kernel_size) / stride_size) + 1;

    }

}


//padded tensor is the final result of the input padding
//input tensor is the returned input tensor from the MNIST_Single_Targeted_Image_To_Matrix function or any function that extracts input
void Tensor_Padding_Symmetric(Tensor &padded_tensor,const Tensor &input_tensor,int pad_top, int pad_bottom,int pad_left, int pad_right) {
    int depth = input_tensor.depth();
    int row   = input_tensor.rows();
    int col   = input_tensor.columns();

    // Create new padded tensor
    int new_rows = row + pad_top + pad_bottom;
    int new_cols = col + pad_left + pad_right;
    padded_tensor = Tensor(new_rows, new_cols, depth);

    // Copy values into the center
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                padded_tensor(r + pad_top, c + pad_left, d) = input_tensor(r, c, d);
            }
        }
    }
}

void Tensor_Apply_Padding_By_Strategy(
        Tensor &padded_tensor,
        const Tensor &input_tensor,
        int kernel_size,
        int stride,
        PaddingStrategy strategy
) {
    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

    if (strategy == SAME) {
        int in_h = input_tensor.rows();
        int in_w = input_tensor.columns();

        int out_h = std::ceil((float)in_h / stride);
        int out_w = std::ceil((float)in_w / stride);

        int total_pad_h = std::max(0, (out_h - 1) * stride + kernel_size - in_h);
        int total_pad_w = std::max(0, (out_w - 1) * stride + kernel_size - in_w);

        pad_top    = total_pad_h / 2;
        pad_bottom = total_pad_h - pad_top;
        pad_left   = total_pad_w / 2;
        pad_right  = total_pad_w - pad_left;
    }

    // Call the core padding function
    Tensor_Padding_Symmetric(padded_tensor, input_tensor, pad_top, pad_bottom, pad_left, pad_right);
}







