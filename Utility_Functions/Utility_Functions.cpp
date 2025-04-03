#include "Utility_Functions.h"
#include "Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include "Neural_Layers/Neural_Layer_Skeleton.h" // For ActivationType

void DisplayNeuralLayerMatrices(const Train_Block_by_Backpropagation& trained_block) {
    try {
        int block_size = trained_block.Get_Block_Size();
        std::vector<Train_Block_by_Backpropagation::layer_information_cache> layer_information =
                trained_block.Get_Layer_Information();

        std::cout << "==== Neural Network Architecture ====" << std::endl;
        std::cout << "Total layers: " << block_size << std::endl << std::endl;

        for (int i = 0; i < block_size; i++) {
            std::cout << "=== Layer " << i << " (" << (i == 0 ? "Input" :
                                                       (i == block_size-1 ? "Output" : "Hidden")) << ") ===" << std::endl;

            // Display matrix dimensions with consistent formatting
            std::cout << "  Input Matrix: [" << layer_information[i].input_matrix.rows()
                      << " X " << layer_information[i].input_matrix.columns() << "]" << std::endl;
            std::cout << "  Weights Matrix: [" << layer_information[i].weights_matrix.rows()
                      << " X  " << layer_information[i].weights_matrix.columns() << "]" << std::endl;
            std::cout << "  Bias Matrix: [" << layer_information[i].bias_matrix.rows()
                      << " X"
                         " " << layer_information[i].bias_matrix.columns() << "]" << std::endl;
            std::cout << "  Pre-Activation Matrix: [" << layer_information[i].pre_activation_tensor.rows()
                      << " X"
                         " " << layer_information[i].pre_activation_tensor.columns() << "]" << std::endl;
            std::cout << "  Post-Activation Matrix: [" << layer_information[i].post_activation_tensor.rows()
                      << " X"
                         " " << layer_information[i].post_activation_tensor.columns() << "]" << std::endl;

            // Add activation type information
            std::cout << "  Activation: " << GetActivationTypeString(layer_information[i].activation_type) << std::endl;
            std::cout << std::endl;

        }
    } catch (const std::exception& e) {
        std::cerr << "Error displaying neural layer matrices: " << e.what() << std::endl;
    }
}

void DisplayBackpropagationGradients(const Train_Block_by_Backpropagation& trained_block) {
    try {
        int block_size = trained_block.Get_Block_Size();
        std::vector<Train_Block_by_Backpropagation::layer_intermediate_cache> intermediate_matrices =
                trained_block.Get_Intermediate_Layer_Information();

        std::cout << "==== Backpropagation Gradient Matrices ====" << std::endl;
        std::cout << "Total layers: " << block_size << std::endl << std::endl;

        for (int i = 0; i < block_size; i++) {
            // Display layer header with appropriate label
            if (i == block_size - 1) {
                std::cout << "=== Output Layer " << i << " Gradients ===" << std::endl;

                // Display additional output layer specific information
                std::cout << "  Predicted Values: [" << intermediate_matrices[i].post_activation_tensor.rows()
                          << " X " << intermediate_matrices[i].post_activation_tensor.columns() << "]" << std::endl;
                std::cout << "  Target Values: [" << intermediate_matrices[i].y_true.rows()
                          << " X " << intermediate_matrices[i].y_true.columns() << "]" << std::endl;
            } else {
                std::cout << "=== Layer " << i << " Gradients ===" << std::endl;
            }

            // Display common gradient information for all layers
            std::cout << "  Activation Derivative (da/dz): [" << intermediate_matrices[i].da_dz.rows()
                      << " X " << intermediate_matrices[i].da_dz.columns() << "]" << std::endl;
            std::cout << "  Loss Gradient wrt Pre-Activation (dL/dz): [" << intermediate_matrices[i].dL_dz.rows()
                      << " X " << intermediate_matrices[i].dL_dz.columns() << "]" << std::endl;
            std::cout << "  Loss Gradient wrt Weights (dL/dW): [" << intermediate_matrices[i].dL_dW.rows()
                      << " X " << intermediate_matrices[i].dL_dW.columns() << "]" << std::endl;
            std::cout << "  Loss Gradient wrt Biases (dL/db): [" << intermediate_matrices[i].dL_db.rows()
                      << " X " << intermediate_matrices[i].dL_db.columns() << "]" << std::endl;

            // Only show dL/da_upper_layer for non-input layers (since it doesn't exist for the input layer)
            if (i > 0) {
                std::cout << "  Loss Gradient for Upper Layer (dL/da): [" << intermediate_matrices[i].dL_da_upper.rows()
                          << " X " << intermediate_matrices[i].dL_da_upper.columns() << "]" << std::endl;
            }

            std::cout << "  Loss Gradient wrt Output (dL/dy): [" << intermediate_matrices[i].dL_dy.rows()
                      << " X " << intermediate_matrices[i].dL_dy.columns() << "]" << std::endl;
            std::cout << "  Transposed Weights Matrix: [" << intermediate_matrices[i].W_transposed.rows()
                      << " X " << intermediate_matrices[i].W_transposed.columns() << "]" << std::endl;
            std::cout << "  Transposed Input Matrix: [" << intermediate_matrices[i].I_transposed.rows()
                      << " X " << intermediate_matrices[i].I_transposed.columns() << "]" << std::endl;

            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error displaying backpropagation gradients: " << e.what() << std::endl;
    }
}



/**
 * @brief Prints a matrix with formatting and statistical insights for debugging
 *
 * @param matrix The matrix to print
 * @param name Name/identifier for the matrix (displayed in header)
 * @param precision Number of decimal places to display (default: 4)
 * @param max_rows Maximum rows to display (0 for all; large matrices will show first/last few rows)
 * @param max_cols Maximum columns to display (0 for all; wide matrices will show first/last few columns)
 */
void Debug_Matrix(const Matrix& matrix, const std::string& name, int precision , int max_rows, int max_cols ) {
    // Matrix dimensions
    int rows = matrix.rows();
    int cols = matrix.columns();

    // Limit rows/columns if needed
    bool truncate_rows = max_rows > 0 && rows > max_rows + 2;
    bool truncate_cols = max_cols > 0 && cols > max_cols + 2;

    int display_rows = truncate_rows ? max_rows : rows;
    int display_cols = truncate_cols ? max_cols : cols;

    // Calculate statistics
    float min_val = 0.0f;
    float max_val = 0.0f;
    float sum = 0.0f;
    float abs_sum = 0.0f;
    int zeros = 0;

// Initialize min_val and max_val only if the matrix is not empty
    if (rows > 0 && cols > 0) {
        min_val = matrix(0, 0);
        max_val = matrix(0, 0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val = matrix(i, j);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
                abs_sum += std::abs(val);
                if (val == 0.0f) zeros++;
            }
        }
    }

    float mean = sum / (rows * cols);

    // Create a header with matrix info
    std::ostringstream header;
    header << "MATRIX: " << name << " [" << rows << "X" << cols << "]";
    std::string header_str = header.str();

    // Print header
    std::cout << "\n" << std::string(header_str.length() + 4, '=') << std::endl;
    std::cout << "| " << header_str << " |" << std::endl;
    std::cout << std::string(header_str.length() + 4, '=') << std::endl;

    // Print stats
    std::cout << std::fixed << std::setprecision(precision);
    std::cout << "Min: " << min_val << " | Max: " << max_val << " | Mean: " << mean
              << " | Sum: " << sum << " | |Sum|: " << abs_sum
              << " | Zeros: " << zeros << "/" << (rows * cols)
              << " (" << (100.0f * zeros / (rows * cols)) << "%)" << std::endl;

    std::cout << std::string(header_str.length() + 4, '-') << std::endl;

    // Determine field width for values
    int field_width = precision + 8; // Allow space for sign, digits, and decimal point

    // Print matrix content
    for (int i = 0; i < rows; ++i) {
        // Skip middle rows if truncating
        if (truncate_rows && i >= display_rows/2 && i < rows - display_rows/2) {
            if (i == display_rows/2) {
                std::cout << std::string(5, ' ') << "..." << std::string(field_width * std::min(cols, 3), ' ') << std::endl;
            }
            continue;
        }

        std::cout << std::setw(3) << i << ": [";

        for (int j = 0; j < cols; ++j) {
            // Skip middle columns if truncating
            if (truncate_cols && j >= display_cols/2 && j < cols - display_cols/2) {
                if (j == display_cols/2) {
                    std::cout << " ... ";
                }
                continue;
            }

            std::cout << std::setw(field_width) << matrix(i, j);

            if (j < cols - 1 && (!truncate_cols || j < display_cols/2-1 || j >= cols - display_cols/2)) {
                std::cout << ",";
            }
        }

        std::cout << " ]" << std::endl;
    }

    std::cout << std::string(header_str.length() + 4, '=') << std::endl;
}

// Helper function to convert activation type to string
std::string GetActivationTypeString(ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::SWISH: return "Swish";
        case ActivationType::LINEAR: return "Linear";
        default: return "Unknown";
    }
}