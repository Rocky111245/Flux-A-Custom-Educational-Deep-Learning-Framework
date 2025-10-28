//
// Created by rakib on 18/5/2025.
//


#include "PoolingLayers.h"


void Pooling_Layer::Initialize_Tensors(){

    //Applies padding and resizes the padded input tensor
    Tensor_Apply_Padding_By_Strategy(padded_input_tensor_,input_tensor_,pool_window_size_,stride_,padding_strategy_);

    // Calculate output dimensions
    int input_rows = input_tensor_.rows();
    int input_columns = input_tensor_.columns();
    int channels = input_tensor_.depth();

    //Find output downsampled feature map size
    int output_rows = Get_Output_Rows(input_rows, pool_window_size_, stride_, padding_strategy_);
    int output_columns = Get_Output_Columns(input_columns, pool_window_size_, stride_, padding_strategy_);

    // Initialize output tensor and indices tensor
    output_tensor_ = Tensor(output_rows, output_columns, channels);

    //This tensor should have the same number of channels as the original/padded input tensor,
    // 2 columns per channel (one for the row indices and one for the column indices)
    //rows should equal the number of times the window slides across the input. An intelligent way to store indices without much hassle.
    if(pooling_strategy_==MAX_POOL){
        max_pool_indices_tensor_ = Tensor(output_rows*output_columns, 2, channels);
    }

}



//Pooling is done on padded input tensor
void Pooling_Layer::Average_Pooling() {
    // Get input dimensions
    int padded_input_rows = padded_input_tensor_.rows();
    int padded_input_columns = padded_input_tensor_.columns();

    // Perform max pooling for all channels
    for(int depth_number = 0; depth_number < padded_input_tensor_.depth(); depth_number++) {
        // Reset indices counter for each channel
        int output_tensor_row_indices = 0;

        // Process each vertical stride position in the global map
        for(int global_vertical_index = 0; global_vertical_index + pool_window_size_ <= padded_input_rows; global_vertical_index += stride_) {
            int output_tensor_column_indices = 0;

            // Process each horizontal stride position in the global map
            for(int global_horizontal_index = 0; global_horizontal_index + pool_window_size_ <= padded_input_columns; global_horizontal_index += stride_) {
                // Reset max value for each window
                float average_value = 0.0f;

                // For loops responsible for scanning the patch
                for(int i = global_vertical_index; i < global_vertical_index + pool_window_size_; i++) {
                    for(int j = global_horizontal_index; j < global_horizontal_index + pool_window_size_; j++) {
                        // Sum up values
                        average_value += padded_input_tensor_(i, j, depth_number);
                    }
                }
                average_value=average_value/(pool_window_size_*pool_window_size_);

                // Store the maximum value in output tensor
                output_tensor_(output_tensor_row_indices, output_tensor_column_indices, depth_number) = average_value;
                output_tensor_column_indices++;
            }
            output_tensor_row_indices++;
        }
    }
}

//Pooling is done on padded input tensor
void Pooling_Layer::Max_Pooling() {
    // Get input dimensions
    int padded_input_rows = padded_input_tensor_.rows();
    int padded_input_columns = padded_input_tensor_.columns();

    // Perform max pooling for all channels
    for(int depth_number = 0; depth_number < padded_input_tensor_.depth(); depth_number++) {
        // Reset indices counter for each channel
        int indices_tensor_index = 0;
        int output_tensor_row_indices = 0;

        // Process each vertical stride position in the global map
        for(int global_vertical_index = 0; global_vertical_index + pool_window_size_ <= padded_input_rows; global_vertical_index += stride_) {
            int output_tensor_column_indices = 0;

            // Process each horizontal stride position in the global map
            for(int global_horizontal_index = 0; global_horizontal_index + pool_window_size_ <= padded_input_columns; global_horizontal_index += stride_) {
                // Reset max value for each window
                float max_value = -std::numeric_limits<float>::max();
                int max_i = -1;
                int max_j = -1;

                // For loops responsible for scanning the patch
                for(int i = global_vertical_index; i < global_vertical_index + pool_window_size_; i++) {
                    for(int j = global_horizontal_index; j < global_horizontal_index + pool_window_size_; j++) {
                        // Check if current value is greater than max found so far
                        float current_value = padded_input_tensor_(i, j, depth_number);
                        if(current_value > max_value) {
                            max_value = current_value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }

                // Store the maximum value in output tensor
                output_tensor_(output_tensor_row_indices, output_tensor_column_indices, depth_number) = max_value;

                max_pool_indices_tensor_(indices_tensor_index, 0, depth_number) = max_i;
                max_pool_indices_tensor_(indices_tensor_index, 1, depth_number) = max_j;


                indices_tensor_index++;
                output_tensor_column_indices++;
            }
            output_tensor_row_indices++;
        }
    }
}

//work in progress
void Pooling_Layer::Backpropagate(Tensor &dL_dx, Tensor &dL_dy){

    //Resize if needed
    if(dL_dx.rows()!=padded_input_tensor_.rows() || dL_dx.columns()!=padded_input_tensor_.columns() || dL_dx.depth()!=padded_input_tensor_.depth()){
        dL_dx = Tensor(padded_input_tensor_.rows(),padded_input_tensor_.columns(),padded_input_tensor_.depth());
    }

    if(pooling_strategy_==MAX_POOL){
for(int d = 0; d < max_pool_indices_tensor_.depth(); d++) {
    for(int r = 0; r < max_pool_indices_tensor_.rows(); r++) {
        for(int c = 0; c < max_pool_indices_tensor_.columns(); c++) {
            if(r!=max_pool_indices_tensor_(r, c, d) != )
            dL_dx(r, c, d) =
        }
    }

}
    }
    else{

    }



}