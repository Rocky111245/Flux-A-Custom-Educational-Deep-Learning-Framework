

#include "Train_Block_by_Backpropagation.h"



Train_Block_by_Backpropagation::Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate )
        : neural_block(neural_block), learning_rate(learning_rate),iterations(iterations) {

    // Initialize data structures
    int block_size = neural_block.Get_Block_Size();
    layer_information.resize(block_size);
    intermediate_matrices.resize(block_size);

    Populate_Layer_Information();

    Initialize_Layer_Intermediate_Matrices();

    Train_by_Backpropagation();

}

// Perform one iteration of backpropagation
void Train_Block_by_Backpropagation::Train_by_Backpropagation() {

    for (int i = 0; i < iterations; ++i) {
        // Perform forward pass
        // Perform forward pass for each iteration
        neural_block.Forward_Pass_With_Activation();


        // Compute gradients
        ComputeAllLayerGradients();

        // Update weights and biases
        UpdateLayerParameters();

        std::cout << "Iteration " << i << ", Loss: " << neural_block.Get_Block_Loss() << std::endl;


    }
}




// Populate layer information from the neural block
void Train_Block_by_Backpropagation::Populate_Layer_Information() {
    int size_of_block = neural_block.Get_Block_Size();
    std::cout<<"Block size: "<<size_of_block<<" "<<std::endl;

    assert(neural_block.Get_Block_Status());

    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {

        //This resizes the left hand matrix to the right hand side (ground truth) if they are not the correct size.
        layer_information[layer_number].input_matrix = neural_block.Get_Block_Layers(layer_number).input_matrix;
        //Debug_Matrix(layer_information[layer_number].input_matrix, "Input Matrix");
        layer_information[layer_number].weights_matrix = neural_block.Get_Block_Layers(layer_number).weights_matrix;
        //Debug_Matrix(layer_information[layer_number].weights_matrix, "Weights Matrix");
        layer_information[layer_number].bias_matrix = neural_block.Get_Block_Layers(layer_number).bias_matrix;
        //Debug_Matrix(layer_information[layer_number].bias_matrix, "Bias Matrix");
        layer_information[layer_number].pre_activation_tensor = neural_block.Get_Block_Layers(layer_number).pre_activation_tensor;
        //Debug_Matrix(layer_information[layer_number].pre_activation_tensor, "Pre Activation Matrix");
        layer_information[layer_number].post_activation_tensor = neural_block.Get_Block_Layers(layer_number).post_activation_tensor;
        //Debug_Matrix(layer_information[layer_number].post_activation_tensor, "Post Activation Matrix");
        layer_information[layer_number].activation_type = neural_block.Get_Block_Layers(layer_number).activationType;

    }
}


// Create intermediate matrices needed for backpropagation
void Train_Block_by_Backpropagation::Initialize_Layer_Intermediate_Matrices() {

    int size_of_block = neural_block.Get_Block_Size();
    int last_layer_number = size_of_block - 1;

    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {

        // Input values reference
        // Forward pass matrices
        intermediate_matrices[layer_number].input_values = layer_information[layer_number].input_matrix;
        intermediate_matrices[layer_number].pre_activation_tensor = layer_information[layer_number].pre_activation_tensor;
        intermediate_matrices[layer_number].post_activation_tensor = layer_information[layer_number].post_activation_tensor;
        intermediate_matrices[layer_number].activation_type = layer_information[layer_number].activation_type;
        std::cout<<"activation type: "<<Neural_Layer_Skeleton::ActivationTypeToString(intermediate_matrices[layer_number].activation_type)<<std::endl;




        // Activation derivatives all set to 0

        //dL_dy matrice = dL_da_upper matrices for output layer
        intermediate_matrices[layer_number].dL_dy = Matrix(
                layer_information[layer_number].post_activation_tensor.rows(),
                layer_information[layer_number].post_activation_tensor.columns(),
                0.0f
        );

        intermediate_matrices[layer_number].da_dz = Matrix(
                layer_information[layer_number].post_activation_tensor.rows(),
                layer_information[layer_number].post_activation_tensor.columns(),
                0.0f
        );

        // Backward pass matrices
        intermediate_matrices[layer_number].dL_dz = Matrix(
                layer_information[layer_number].pre_activation_tensor.rows(),
                layer_information[layer_number].pre_activation_tensor.columns(),
                0.0f
        );

        intermediate_matrices[layer_number].dL_dW = Matrix(
                layer_information[layer_number].weights_matrix.rows(),
                layer_information[layer_number].weights_matrix.columns(),
                0.0f
        );

        // Bias gradients have the same matrix as dL_dz
        intermediate_matrices[layer_number].dL_db = Matrix(
                layer_information[layer_number].bias_matrix.rows(),
                layer_information[layer_number].bias_matrix.columns(),
                0.0f
        );

        //This gradient goes to the previous layer. dL_da_upper= dL_dz* dz_da

        if(layer_number>0){
            intermediate_matrices[layer_number-1].dL_da_upper = Matrix(
                    layer_information[layer_number].input_matrix.rows(),
                    layer_information[layer_number].input_matrix.columns(),
                    0.0f
            );
        }


        // Cached matrices

        intermediate_matrices[layer_number].W= Matrix(
                layer_information[layer_number].weights_matrix.rows(),
                layer_information[layer_number].weights_matrix.columns(),
                0.0f
        );
        intermediate_matrices[layer_number].W= layer_information[layer_number].weights_matrix;

        // Transpose matrices
        intermediate_matrices[layer_number].W_transposed= Matrix(
                layer_information[layer_number].weights_matrix.columns(),
                layer_information[layer_number].weights_matrix.rows()
                );


        Matrix_Transpose(intermediate_matrices[layer_number].W_transposed, layer_information[layer_number].weights_matrix);


        intermediate_matrices[layer_number].I_transposed= Matrix(
                layer_information[layer_number].input_matrix.columns(),
                layer_information[layer_number].input_matrix.rows()
                );

        Matrix_Transpose(intermediate_matrices[layer_number].I_transposed, layer_information[layer_number].input_matrix);


    }
    if(size_of_block>0){
        intermediate_matrices[last_layer_number].y_pred = intermediate_matrices[last_layer_number].post_activation_tensor;
        intermediate_matrices[last_layer_number].y_true = neural_block.Get_Block_Target_Matrix();
    }
}

// Compute gradients for the output layer
void Train_Block_by_Backpropagation::ComputeAllLayerGradients() {

    int size_of_block = neural_block.Get_Block_Size();
    int output_layer = size_of_block - 1;

    for(int i = 0; i < size_of_block; ++i){
//        std::cout<<"After Forward Pass,one Round, Layer:  "<<i<<std::endl;
//        Debug_Matrix(intermediate_matrices[i].input_values , "Input Matrix");
//        Debug_Matrix(intermediate_matrices[i].W, "Weights Matrix");
//        Debug_Matrix(intermediate_matrices[i].W_transposed, "Weights Transposed Matrix");
//        Debug_Matrix(intermediate_matrices[i].I_transposed, "Input Transposed Matrix");
//        Debug_Matrix(intermediate_matrices[i].pre_activation_tensor, "Pre Activation Matrix");
//        Debug_Matrix(intermediate_matrices[i].post_activation_tensor, "Post Activation Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dy, "dL_dy Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dz, "dL_dz Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dW, "dL_dW Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_db, "dL_db Matrix");
//        Debug_Matrix(intermediate_matrices[i].da_dz, "da_dz Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_da_upper, "dL_da_upper Matrix");
    }


    // Start with output layer
    // Calculate gradient of loss with respect to the output.dL_dy = dL_da (output layer only)
    CalculateLossGradient(
            intermediate_matrices[output_layer].dL_dy,
            intermediate_matrices[output_layer].y_pred,
            intermediate_matrices[output_layer].y_true
    );

    // Process all layers from output to input
    for(int i = output_layer; i >= 0; i--) {


        // 1. Calculate activation function derivative da_dz
        CalculateActivationDerivative(
                intermediate_matrices[i].da_dz,
                intermediate_matrices[i].pre_activation_tensor,
                intermediate_matrices[i].activation_type
        );

        // 2. Calculate gradient of loss with respect to pre-activation.
        if(i == output_layer) {
            assert(Matrix_Can_HadamardProduct(intermediate_matrices[i].dL_dz,
                                              intermediate_matrices[i].dL_dy,
                                              intermediate_matrices[i].da_dz));
            Matrix_Hadamard_Product(
                    intermediate_matrices[i].dL_dz,
                    intermediate_matrices[i].dL_dy,
                    intermediate_matrices[i].da_dz
            );
        } else {
            assert(Matrix_Can_HadamardProduct(intermediate_matrices[i].dL_dz,
                                              intermediate_matrices[i].dL_da_upper,
                                              intermediate_matrices[i].da_dz));
            Matrix_Hadamard_Product(
                    intermediate_matrices[i].dL_dz,
                    intermediate_matrices[i].dL_da_upper,
                    intermediate_matrices[i].da_dz
            );

        }

        // 3. Calculate gradients with respect to weights and biases. dl_dW has the same shape as the weight matrix.However, during multiplication, a different order is required. So temp was created.
        Matrix temp_dL_dW(intermediate_matrices[i].dL_dW.columns(), intermediate_matrices[i].dL_dW.rows());// to accomodate the transpose
        assert(Matrix_Can_Multiply(temp_dL_dW,
                                   intermediate_matrices[i].I_transposed,
                                   intermediate_matrices[i].dL_dz));
        Matrix_Multiply(temp_dL_dW, intermediate_matrices[i].I_transposed, intermediate_matrices[i].dL_dz);  // (7×1)
        Matrix_Transpose(intermediate_matrices[i].dL_dW, temp_dL_dW);               // Now (1×7), matches the W shape





        // Later, after dL_dz has been calculated,we find the gradient of the loss with respect to the biases.
        Matrix_Sum_Columns_To_One_Row(intermediate_matrices[i].dL_db, intermediate_matrices[i].dL_dz);


        // 4. Propagate gradients to previous layer if it exists. dl_da_upper has the same shape as the input matrix to this layer.However, during multiplication, a different order is required. So temp was created.
        if (i > 0) {

            assert(Matrix_Can_Multiply(intermediate_matrices[i-1].dL_da_upper,
                                       intermediate_matrices[i].dL_dz,
                                       intermediate_matrices[i].W));
            Matrix_Multiply(
                    intermediate_matrices[i-1].dL_da_upper,
                    intermediate_matrices[i].dL_dz,
                    intermediate_matrices[i].W
            );

        }
//        std::cout<<"After Gradient Calculations,One Round, Layer:  "<<i<<std::endl;
//
//        Debug_Matrix(intermediate_matrices[i].input_values , "Input Matrix");
//        Debug_Matrix(intermediate_matrices[i].W, "Weights Matrix");
//        Debug_Matrix(intermediate_matrices[i].W_transposed, "Weights Transposed Matrix");
//        Debug_Matrix(intermediate_matrices[i].I_transposed, "Input Transposed Matrix");
//        Debug_Matrix(intermediate_matrices[i].pre_activation_tensor, "Pre Activation Matrix");
//        Debug_Matrix(intermediate_matrices[i].post_activation_tensor, "Post Activation Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dy, "dL_dy Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dz, "dL_dz Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_dW, "dL_dW Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_db, "dL_db Matrix");
//        Debug_Matrix(intermediate_matrices[i].da_dz, "da_dz Matrix");
//        Debug_Matrix(intermediate_matrices[i].dL_da_upper, "dL_da_upper Matrix");
    }
}



// Update weights and biases using calculated gradients
void Train_Block_by_Backpropagation::UpdateLayerParameters() {
    int size_of_block = neural_block.Get_Block_Size();

    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {
        // Create references for better readability
        Matrix& weights = layer_information[layer_number].weights_matrix;
        Matrix& biases = layer_information[layer_number].bias_matrix;
        const Matrix& weight_gradients = intermediate_matrices[layer_number].dL_dW;
        const Matrix& bias_gradients = intermediate_matrices[layer_number].dL_db;

        // Update weights directly (avoiding temporary matrix creation)
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.columns(); ++j) {
                weights(i, j) -= learning_rate * weight_gradients(i, j);
            }
        }

        // Update biases directly
        for (int i = 0; i < biases.rows(); ++i) {
            for (int j = 0; j < biases.columns(); ++j) {
                biases(i, j) -= learning_rate * bias_gradients(i, j);
            }
        }

        // Transfer updated parameters to neural block
        neural_block.Set_Layers(layer_number).weights_matrix = weights;
        neural_block.Set_Layers(layer_number).bias_matrix = biases;
    }

}



// Add this to the public methods section of Neural_Block class
Matrix Train_Block_by_Backpropagation::Get_Predictions() const {
    // Return post-activation tensor of the last layer
    int last_layer = neural_block.Get_Block_Size() - 1;
    return neural_block.Get_Block_Layers(last_layer).post_activation_tensor;
}





// Calculate loss gradient based on the loss function
void Train_Block_by_Backpropagation::CalculateLossGradient( Matrix& gradient,const Matrix& predicted, const Matrix& target) {
    LossFunction loss_function = neural_block.Get_Block_Loss_Type();

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
            const float EPSILON = 1e-7f;
            for (int i = 0; i < predicted.rows(); ++i) {
                for (int j = 0; j < predicted.columns(); ++j) {
                    float pred = std::clamp(predicted(i, j), EPSILON, 1.0f - EPSILON);
                    gradient(i, j) = (pred - target(i, j)) / (pred * (1.0f - pred) * predicted.rows());
                }
            }
            break;
        }

        case LossFunction::HUBER_LOSS: {
            // For Huber Loss
            const float delta = 1.0f; // Default threshold
            for (int i = 0; i < predicted.rows(); ++i) {
                for (int j = 0; j < predicted.columns(); ++j) {
                    float error = predicted(i, j) - target(i, j);
                    if (std::abs(error) <= delta) {
                        gradient(i, j) = error / predicted.rows();
                    } else {
                        gradient(i, j) = (error > 0 ? 1.0f : -1.0f) * delta / predicted.rows();
                    }
                }
            }
            break;
        }

        case LossFunction::MAE_LOSS: {

            // The derivative of |x| is sign(x): 1 for x > 0, -1 for x < 0, 0 for x = 0
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

// Calculate derivatives of activation functions
void Train_Block_by_Backpropagation::CalculateActivationDerivative(Matrix& derivatives,const Matrix& z,ActivationType activation_type) {
    if (z.rows() != derivatives.rows() || z.columns() != derivatives.columns()) {
        throw std::invalid_argument("Input and output matrices must have the same dimensions.");
    }

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
            // Leaky ReLU derivative: f'(x) = 1 if x > 0, else alpha
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



std::vector<Train_Block_by_Backpropagation::layer_information_cache> Train_Block_by_Backpropagation::Get_Layer_Information() const {
    return layer_information;
}

std::vector<Train_Block_by_Backpropagation::layer_intermediate_cache> Train_Block_by_Backpropagation::Get_Intermediate_Layer_Information() const {
    return intermediate_matrices;
}

int Train_Block_by_Backpropagation::Get_Block_Size() const{
    return neural_block.Get_Block_Size();
}


// Function to sum columns of a matrix and store in the destination matrix
void Train_Block_by_Backpropagation::Matrix_Sum_Columns_To_One_Row(Matrix& dest, const Matrix& src) {
    if (&dest == &src) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (dest.columns() != src.columns()) {
        throw std::invalid_argument("Destination matrix must have the same number of columns as the source matrix.");
    }

    if (dest.rows() != 1) {
        throw std::invalid_argument("Destination matrix must have exactly 1 row for column summation.");
    }

    for (int col = 0; col < src.columns(); ++col) {
        float column_sum = 0;
        for (int row = 0; row < src.rows(); ++row) {
            column_sum += src(row, col);
        }
        dest(0, col) = column_sum;
    }
}



