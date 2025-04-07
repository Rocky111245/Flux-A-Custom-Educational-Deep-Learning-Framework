/**
 * @file Train_Block_by_Backpropagation.cpp
 * @brief Implementation of backpropagation training algorithm for neural networks
 * @author Rakib
 * @date 2025-03-28
 */

#include "Train_Block_by_Backpropagation.h"

Train_Block_by_Backpropagation::Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate)
        : neural_block(neural_block), learning_rate(learning_rate), iterations(iterations) {
    // Initialize data structures with proper sizes based on network architecture
    int block_size = neural_block.Get_Block_Size();
    intermediate_matrices.resize(block_size);

    // Set up all matrices needed for backpropagation
    Initialize_Layer_Intermediate_Matrices();

    // Start training immediately upon construction
    Train_by_Backpropagation();
}

void Train_Block_by_Backpropagation::Train_by_Backpropagation() {
    // Training loop for the specified number of iterations
    for (int i = 0; i < iterations; ++i) {
        // Step 1: Forward pass - compute outputs for all layers
        neural_block.Forward_Pass_With_Activation();

        // Step 2: Backward pass - compute gradients for all parameters
        ComputeAllLayerGradients();

        // Step 3: Update parameters using computed gradients
        UpdateLayerParameters();

        // Log progress after each iteration
        std::cout << "Iteration " << i << ", Loss: " << neural_block.Get_Block_Loss() << std::endl;
    }
}

void Train_Block_by_Backpropagation::Initialize_Layer_Intermediate_Matrices() {
    int size_of_block = neural_block.Get_Block_Size();
    int last_layer_number = size_of_block - 1;

    // Initialize matrices for each layer
    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {
        // Store references to the neural block's matrices for direct updates
        intermediate_matrices[layer_number].input_values = &neural_block.Set_Block_Layers(layer_number).input_matrix;
        intermediate_matrices[layer_number].weights_matrix = &neural_block.Set_Block_Layers(layer_number).weights_matrix;
        intermediate_matrices[layer_number].bias_matrix = &neural_block.Set_Block_Layers(layer_number).bias_matrix;
        intermediate_matrices[layer_number].pre_activation_tensor = &neural_block.Set_Block_Layers(layer_number).pre_activation_tensor;
        intermediate_matrices[layer_number].post_activation_tensor = &neural_block.Set_Block_Layers(layer_number).post_activation_tensor;
        intermediate_matrices[layer_number].activation_type = neural_block.Set_Block_Layers(layer_number).activationType;

        // Log activation type for debugging
        std::cout << "activation type: " << Neural_Layer_Skeleton::ActivationTypeToString(intermediate_matrices[layer_number].activation_type) << std::endl;

        // Special handling for output layer
        intermediate_matrices[last_layer_number].y_pred = &neural_block.Set_Block_Layers(last_layer_number).post_activation_tensor;
        intermediate_matrices[last_layer_number].y_true = neural_block.Get_Block_Target_Matrix();

        // Initialize gradient matrices for loss w.r.t. activations
        intermediate_matrices[layer_number].dL_dy = Matrix(
                neural_block.Set_Block_Layers(layer_number).post_activation_tensor.rows(),
                neural_block.Set_Block_Layers(layer_number).post_activation_tensor.columns(),
                0.0f
        );

        // Initialize activation derivatives matrix
        intermediate_matrices[layer_number].da_dz = Matrix(
                neural_block.Set_Block_Layers(layer_number).post_activation_tensor.rows(),
                neural_block.Set_Block_Layers(layer_number).post_activation_tensor.columns(),
                0.0f
        );

        // Initialize gradient matrices for pre-activation values
        intermediate_matrices[layer_number].dL_dz = Matrix(
                neural_block.Set_Block_Layers(layer_number).pre_activation_tensor.rows(),
                neural_block.Set_Block_Layers(layer_number).pre_activation_tensor.columns(),
                0.0f
        );

        // Initialize gradient matrices for weights
        intermediate_matrices[layer_number].dL_dW = Matrix(
                neural_block.Set_Block_Layers(layer_number).weights_matrix.rows(),
                neural_block.Set_Block_Layers(layer_number).weights_matrix.columns(),
                0.0f
        );

        // Initialize gradient matrices for biases
        intermediate_matrices[layer_number].dL_db = Matrix(
                neural_block.Set_Block_Layers(layer_number).bias_matrix.rows(),
                neural_block.Set_Block_Layers(layer_number).bias_matrix.columns(),
                0.0f
        );

        // Initialize gradients for propagation to previous layer
        if (layer_number > 0) {
            intermediate_matrices[layer_number-1].dL_da_upper = Matrix(
                    neural_block.Set_Block_Layers(layer_number).input_matrix.rows(),
                    neural_block.Set_Block_Layers(layer_number).input_matrix.columns(),
                    0.0f
            );
        }

        // Initialize cached transpose matrices
        intermediate_matrices[layer_number].W_transposed = Matrix(
                neural_block.Set_Block_Layers(layer_number).weights_matrix.columns(),
                neural_block.Set_Block_Layers(layer_number).weights_matrix.rows(),
                0.0f
        );

        intermediate_matrices[layer_number].I_transposed = Matrix(
                neural_block.Set_Block_Layers(layer_number).input_matrix.columns(),
                neural_block.Set_Block_Layers(layer_number).input_matrix.rows(),
                0.0f
        );
    }
}

void Train_Block_by_Backpropagation::ComputeAllLayerGradients() {
    int size_of_block = neural_block.Get_Block_Size();
    int output_layer = size_of_block - 1;

    // Start with output layer gradient calculation
    CalculateLossGradient(
            intermediate_matrices[output_layer].dL_dy,
            *intermediate_matrices[output_layer].y_pred,
            intermediate_matrices[output_layer].y_true
    );

    // Process all layers from output to input (backpropagation)
    for(int i = output_layer; i >= 0; i--) {
        // Update cached transposed matrices for efficient calculations
        Matrix_Transpose(intermediate_matrices[i].W_transposed, *intermediate_matrices[i].weights_matrix);
        Matrix_Transpose(intermediate_matrices[i].I_transposed, *intermediate_matrices[i].input_values);

        // Step 1: Calculate activation function derivative (da/dz)
        CalculateActivationDerivative(
                intermediate_matrices[i].da_dz,
                *intermediate_matrices[i].pre_activation_tensor,
                intermediate_matrices[i].activation_type
        );

        // Step 2: Calculate gradient of loss w.r.t. pre-activation (dL/dz)
        if(i == output_layer) {
            // For output layer: dL/dz = dL/dy * da/dz
            assert(Matrix_Can_HadamardProduct(intermediate_matrices[i].dL_dz,
                                              intermediate_matrices[i].dL_dy,
                                              intermediate_matrices[i].da_dz));
            Matrix_Hadamard_Product(
                    intermediate_matrices[i].dL_dz,
                    intermediate_matrices[i].dL_dy,
                    intermediate_matrices[i].da_dz
            );
        } else {
            // For hidden layers: dL/dz = dL/da * da/dz
            assert(Matrix_Can_HadamardProduct(intermediate_matrices[i].dL_dz,
                                              intermediate_matrices[i].dL_da_upper,
                                              intermediate_matrices[i].da_dz));
            Matrix_Hadamard_Product(
                    intermediate_matrices[i].dL_dz,
                    intermediate_matrices[i].dL_da_upper,
                    intermediate_matrices[i].da_dz
            );
        }

        // Step 3: Calculate gradient of loss w.r.t. weights (dL/dW)
        // Create temporary matrix for transposition
        Matrix temp_dL_dW(intermediate_matrices[i].dL_dW.columns(), intermediate_matrices[i].dL_dW.rows());

        // Calculate dL/dW = X^T * dL/dz (using proper dimensions)
        assert(Matrix_Can_Multiply(temp_dL_dW,
                                   intermediate_matrices[i].I_transposed,
                                   intermediate_matrices[i].dL_dz));
        Matrix_Multiply(temp_dL_dW, intermediate_matrices[i].I_transposed, intermediate_matrices[i].dL_dz);

        // Transpose to match weight matrix dimensions
        Matrix_Transpose(intermediate_matrices[i].dL_dW, temp_dL_dW);

        // Step 4: Calculate gradient of loss w.r.t. biases (dL/db)
        Matrix_Sum_Columns_To_One_Row(intermediate_matrices[i].dL_db, intermediate_matrices[i].dL_dz);

        // Step 5: Propagate gradients to previous layer if it exists
        if (i > 0) {
            // dL/da_prev = dL/dz * W
            assert(Matrix_Can_Multiply(intermediate_matrices[i-1].dL_da_upper,
                                       intermediate_matrices[i].dL_dz,
                                       *intermediate_matrices[i].weights_matrix));
            Matrix_Multiply(
                    intermediate_matrices[i-1].dL_da_upper,
                    intermediate_matrices[i].dL_dz,
                    *intermediate_matrices[i].weights_matrix
            );
        }
    }
}

void Train_Block_by_Backpropagation::UpdateLayerParameters() {
    int size_of_block = neural_block.Get_Block_Size();

    // Update parameters for each layer using computed gradients
    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {
        // Create references for better readability
        Matrix& weights = *intermediate_matrices[layer_number].weights_matrix;
        Matrix& biases = *intermediate_matrices[layer_number].bias_matrix;
        const Matrix& weight_gradients = intermediate_matrices[layer_number].dL_dW;
        const Matrix& bias_gradients = intermediate_matrices[layer_number].dL_db;

        // Update weights with gradient descent: W = W - learning_rate * dL/dW
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.columns(); ++j) {
                weights(i, j) -= learning_rate * weight_gradients(i, j);
            }
        }

        // Update biases with gradient descent: b = b - learning_rate * dL/db
        for (int i = 0; i < biases.rows(); ++i) {
            for (int j = 0; j < biases.columns(); ++j) {
                biases(i, j) -= learning_rate * bias_gradients(i, j);
            }
        }
    }
}

Matrix Train_Block_by_Backpropagation::Get_Predictions() const {
    // Return post-activation tensor of the last layer (network output)
    int last_layer = neural_block.Get_Block_Size() - 1;
    return neural_block.Set_Block_Layers(last_layer).post_activation_tensor;
}

void Train_Block_by_Backpropagation::CalculateLossGradient(Matrix& gradient, const Matrix& predicted, const Matrix& target) {
    // Get the loss function type from the neural block
    LossFunction loss_function = neural_block.Get_Block_Loss_Type();

    // Calculate gradient based on the specific loss function
    switch (loss_function) {
        case LossFunction::MSE_LOSS: {
            // For MSE: dL/dy = 2(predicted - target)/n
            Matrix diff(predicted.rows(), predicted.columns(), 0.0f);
            assert(Matrix_Can_AddOrSubtract(diff, predicted, target));
            Matrix_Subtract(diff, predicted, target);
            Matrix_Scalar_Multiply(diff, 2.0f / predicted.rows());

            gradient = diff;
            break;
        }

        case LossFunction::CROSS_ENTROPY_LOSS: {
            // For Binary Cross Entropy: dL/dy = (predicted - target)/(predicted*(1-predicted))/n
            const float EPSILON = 1e-7f;  // Small constant to avoid division by zero
            for (int i = 0; i < predicted.rows(); ++i) {
                for (int j = 0; j < predicted.columns(); ++j) {
                    float pred = std::clamp(predicted(i, j), EPSILON, 1.0f - EPSILON);
                    gradient(i, j) = (pred - target(i, j)) / (pred * (1.0f - pred) * predicted.rows());
                }
            }
            break;
        }

        case LossFunction::HUBER_LOSS: {
            // For Huber Loss: combination of MSE and MAE
            const float delta = 1.0f;  // Default threshold parameter
            for (int i = 0; i < predicted.rows(); ++i) {
                for (int j = 0; j < predicted.columns(); ++j) {
                    float error = predicted(i, j) - target(i, j);
                    if (std::abs(error) <= delta) {
                        // Quadratic region: gradient = error/n
                        gradient(i, j) = error / predicted.rows();
                    } else {
                        // Linear region: gradient = sign(error)*delta/n
                        gradient(i, j) = (error > 0 ? 1.0f : -1.0f) * delta / predicted.rows();
                    }
                }
            }
            break;
        }

        case LossFunction::MAE_LOSS: {
            // For MAE: dL/dy = sign(predicted - target)/n
            int num_samples = predicted.rows();
            for (int i = 0; i < predicted.rows(); ++i) {
                for (int j = 0; j < predicted.columns(); ++j) {
                    float diff = predicted(i, j) - target(i, j);
                    if (diff > 0) {
                        gradient(i, j) = 1.0f / num_samples;
                    } else if (diff < 0) {
                        gradient(i, j) = -1.0f / num_samples;
                    } else {
                        gradient(i, j) = 0.0f;  // At exactly zero, the derivative is undefined
                    }
                }
            }
            break;
        }

        default:
            throw std::invalid_argument("Unsupported loss function for gradient calculation.");
    }
}

void Train_Block_by_Backpropagation::CalculateActivationDerivative(Matrix& derivatives, const Matrix& z, ActivationType activation_type) {
    // Validate matrix dimensions
    if (z.rows() != derivatives.rows() || z.columns() != derivatives.columns()) {
        throw std::invalid_argument("Input and output matrices must have the same dimensions.");
    }

    // Calculate derivatives based on activation function type
    switch (activation_type) {
        case ActivationType::RELU: {
            // ReLU derivative: f'(x) = 1 if x > 0, else 0
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    derivatives(i, j) = (z(i, j) > 0) ? 1.0f : 0.0f;
                }
            }
            break;
        }

        case ActivationType::SIGMOID: {
            // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    float sigmoid_value = 1.0f / (1.0f + std::exp(-z(i, j)));
                    derivatives(i, j) = sigmoid_value * (1.0f - sigmoid_value);
                }
            }
            break;
        }

        case ActivationType::TANH: {
            // Tanh derivative: f'(x) = 1 - f(x)²
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    float tanh_value = std::tanh(z(i, j));
                    derivatives(i, j) = 1.0f - (tanh_value * tanh_value);
                }
            }
            break;
        }

        case ActivationType::LEAKY_RELU: {
            // Leaky ReLU derivative: f'(x) = 1 if x > 0, else alpha (small positive value)
            const float alpha = 0.01f;
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    derivatives(i, j) = (z(i, j) > 0) ? 1.0f : alpha;
                }
            }
            break;
        }

        case ActivationType::SWISH: {
            // Swish derivative: f'(x) = f(x) + sigmoid(x)*(1 - f(x))
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    float x = z(i, j);
                    float sigmoid_value = 1.0f / (1.0f + std::exp(-x));
                    float swish_value = x * sigmoid_value;
                    derivatives(i, j) = sigmoid_value + x * sigmoid_value * (1.0f - sigmoid_value);
                }
            }
            break;
        }

        case ActivationType::LINEAR: {
            // Linear derivative: f'(x) = 1
            for (int i = 0; i < z.rows(); ++i) {
                for (int j = 0; j < z.columns(); ++j) {
                    derivatives(i, j) = 1.0f;
                }
            }
            break;
        }

        default:
            throw std::invalid_argument("Invalid activation type for derivative calculation.");
    }
}

std::vector<Train_Block_by_Backpropagation::layer_intermediate_cache> Train_Block_by_Backpropagation::Get_Intermediate_Layer_Information() const {
    return intermediate_matrices;
}

int Train_Block_by_Backpropagation::Get_Block_Size() const {
    return neural_block.Get_Block_Size();
}

void Train_Block_by_Backpropagation::Matrix_Sum_Columns_To_One_Row(Matrix& dest, const Matrix& src) {
    // Validate matrices
    if (&dest == &src) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (dest.columns() != src.columns()) {
        throw std::invalid_argument("Destination matrix must have the same number of columns as the source matrix.");
    }

    if (dest.rows() != 1) {
        throw std::invalid_argument("Destination matrix must have exactly 1 row for column summation.");
    }

    // Sum each column into the single row of the destination matrix
    for (int col = 0; col < src.columns(); ++col) {
        float column_sum = 0;
        for (int row = 0; row < src.rows(); ++row) {
            column_sum += src(row, col);
        }
        dest(0, col) = column_sum;
    }
}