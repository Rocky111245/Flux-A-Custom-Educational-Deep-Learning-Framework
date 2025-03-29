

#include "Train_Block_by_Backpropagation.h"



Train_Block_by_Backpropagation::Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate )
        : neural_block(neural_block), learning_rate(learning_rate),iterations(iterations) {

    // Initialize data structures
    int block_size = neural_block.Get_Block_Size();
    layer_information.resize(block_size);
    intermediate_matrices.resize(block_size);

    Populate_Layer_Information();

    Create_Layer_Intermediate_Matrices();

    Train_by_Backpropagation();

}

// Perform one iteration of backpropagation
void Train_Block_by_Backpropagation::Train_by_Backpropagation() {

    DisplayNeuralLayerMatrices(*this);
    DisplayBackpropagationGradients(*this);
    for (int i = 0; i < iterations; ++i) {
        // Perform forward pass
        // Perform forward pass for each iteration
        neural_block.Forward_Pass_With_Activation();


        // Compute gradients
        ComputeOutputLayerGradients();
        ComputeHiddenLayerGradients();

        // Update weights and biases
        UpdateLayerParameters();
        UpdateLayerInformation();

        std::cout << "Iteration " << i << ", Loss: " << cost << std::endl;


    }
}




// Populate layer information from the neural block
void Train_Block_by_Backpropagation::Populate_Layer_Information() {
    int size_of_block = neural_block.Get_Block_Size();
    std::cout<<"Block size: "<<size_of_block<<" "<<std::endl;

    assert(neural_block.Get_Block_Status());

    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {

        //for last layer
        if (layer_number == size_of_block-1) {
            neural_block.Calculate_Block_Loss();
            cost = neural_block.Get_Loss();
            std::cout<<"Initial Cost: "<<cost<<std::endl;

        }

        //This resizes the left hand matrix to the right hand side (ground truth) if they are not the correct size.
        layer_information[layer_number].input_matrix = neural_block.Get_Layers(layer_number).input_matrix;
        layer_information[layer_number].weights_matrix = neural_block.Get_Layers(layer_number).weights_matrix;
        layer_information[layer_number].bias_matrix = neural_block.Get_Layers(layer_number).bias_matrix;
        layer_information[layer_number].pre_activation_tensor = neural_block.Get_Layers(layer_number).pre_activation_tensor;
        layer_information[layer_number].post_activation_tensor = neural_block.Get_Layers(layer_number).post_activation_tensor;


        layer_information[layer_number].activation_type = neural_block.Get_Layers(layer_number).activationType;
        assert(layer_information[layer_number].activation_type == neural_block.Get_Layers(layer_number).activationType);


    }
}

// Create intermediate matrices needed for backpropagation
void Train_Block_by_Backpropagation::Create_Layer_Intermediate_Matrices() {

    int size_of_block = neural_block.Get_Block_Size();

    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {

        // Input values reference
        intermediate_matrices[layer_number].input_values = layer_information[layer_number].input_matrix;



        // Forward pass matrices
        intermediate_matrices[layer_number].pre_activation_tensor = layer_information[layer_number].pre_activation_tensor;

        intermediate_matrices[layer_number].post_activation_tensor = layer_information[layer_number].post_activation_tensor;

        intermediate_matrices[layer_number].activation_type = layer_information[layer_number].activation_type;
        std::cout<<"activation type: "<<Neural_Layer_Skeleton::ActivationTypeToString(intermediate_matrices[layer_number].activation_type)<<std::endl;

        // Activation derivatives all set to 0
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

        intermediate_matrices[layer_number].dL_db = Matrix(
                layer_information[layer_number].bias_matrix.rows(),
                layer_information[layer_number].bias_matrix.columns(),
                0.0f
        );

        intermediate_matrices[layer_number].dL_da_upper_layer = Matrix(
                layer_information[layer_number].input_matrix.rows(),
                layer_information[layer_number].input_matrix.columns(),
                0.0f
        );

        intermediate_matrices[layer_number].dL_dy = Matrix(
                layer_information[layer_number].post_activation_tensor.rows(),
                layer_information[layer_number].post_activation_tensor.columns(),
                0.0f
        );


        // Cached inverted matrices
        intermediate_matrices[layer_number].W_transposed= Matrix(layer_information[layer_number].weights_matrix.columns(),layer_information[layer_number].weights_matrix.rows());
        intermediate_matrices[layer_number].I_transposed= Matrix(layer_information[layer_number].input_matrix.columns(),layer_information[layer_number].input_matrix.rows());
        Matrix_Transpose(intermediate_matrices[layer_number].W_transposed, layer_information[layer_number].weights_matrix);
        Matrix_Transpose(intermediate_matrices[layer_number].I_transposed, layer_information[layer_number].input_matrix);
    }
}

// Compute gradients for the output layer
void Train_Block_by_Backpropagation::ComputeOutputLayerGradients() {

    int size_of_block = neural_block.Get_Block_Size();
    // Get the last layer
    int last_layer_number= size_of_block - 1;

    output_layer_matrices.y_pred = layer_information[last_layer_number].post_activation_tensor;
    output_layer_matrices.y_true = neural_block.Get_Output_Matrix();

    // 1. Calculate gradient of loss with respect to the output (dL/dy)
    CalculateLossGradient( intermediate_matrices[last_layer_number].dL_dy,output_layer_matrices.y_pred ,output_layer_matrices.y_true);

    // 2. Calculate activation function derivative
    CalculateActivationDerivative(
            intermediate_matrices[last_layer_number].da_dz,
            intermediate_matrices[last_layer_number].pre_activation_tensor,
            intermediate_matrices[last_layer_number].activation_type
    );

    // 3. Calculate gradient of loss with respect to pre-activation (dL/dz)
    // dL/dz = dL/dy * g'(z)
    assert(Matrix_Can_HadamardProduct(intermediate_matrices[last_layer_number].dL_dz,
                                      intermediate_matrices[last_layer_number].dL_dy,
                                      intermediate_matrices[last_layer_number].da_dz));
    Matrix_Hadamard_Product(
            intermediate_matrices[last_layer_number].dL_dz,
            intermediate_matrices[last_layer_number].dL_dy,
            intermediate_matrices[last_layer_number].da_dz
    );


    // 4. Calculate gradient of loss with respect to weights (dL/dW)


    // dL/dW = (input)^T * dL/dz
    if(intermediate_matrices[last_layer_number].dL_dW.rows()==layer_information[last_layer_number].weights_matrix.rows() &&
       intermediate_matrices[last_layer_number].dL_dW.columns()==layer_information[last_layer_number].weights_matrix.columns()){
        Matrix_Resize(intermediate_matrices[last_layer_number].dL_dW,layer_information[last_layer_number].weights_matrix.columns(),
                      layer_information[last_layer_number].weights_matrix.rows());//this is just to resize the matrix so that it can
        // be multiplied
    }
    assert(Matrix_Can_Multiply(intermediate_matrices[last_layer_number].dL_dW, intermediate_matrices[last_layer_number].I_transposed, intermediate_matrices[last_layer_number].dL_dz));
    Matrix_Multiply(intermediate_matrices[last_layer_number].dL_dW, intermediate_matrices[last_layer_number].I_transposed, intermediate_matrices[last_layer_number].dL_dz);
    Matrix temp=intermediate_matrices[last_layer_number].dL_dW;
    Matrix_Resize(intermediate_matrices[last_layer_number].dL_dW,layer_information[last_layer_number].weights_matrix.rows(),
                  layer_information[last_layer_number].weights_matrix.columns());
    Matrix_Transpose(intermediate_matrices[last_layer_number].dL_dW, temp);


    // 5. Calculate gradient of loss with respect to biases (dL/db)
    // dL/db = dL/dz
    intermediate_matrices[last_layer_number].dL_db=intermediate_matrices[last_layer_number].dL_dz;

}

// Compute gradients for hidden layers
void Train_Block_by_Backpropagation::ComputeHiddenLayerGradients() {
    int size_of_block= neural_block.Get_Block_Size();

    // 1. Get gradient from the output layer (dL/da)
    for(int layer_number = size_of_block - 2; layer_number >= 0; --layer_number) {

        CalculateActivationDerivative(
                // 2. Calculate activation function derivative
                intermediate_matrices[layer_number].da_dz,
                intermediate_matrices[layer_number].pre_activation_tensor,
                intermediate_matrices[layer_number].activation_type
        );

        // 3. Calculate gradient of loss with respect to pre-activation (dL/dz)
        // dL/dz = dL/da * g'(z)


        assert(Matrix_Can_HadamardProduct(intermediate_matrices[layer_number].dL_dz,
                                          intermediate_matrices[layer_number].dL_da_upper_layer,
                                          intermediate_matrices[layer_number].da_dz));
        Matrix_Hadamard_Product(
                intermediate_matrices[layer_number].dL_dz,
                intermediate_matrices[layer_number].dL_da_upper_layer,
                intermediate_matrices[layer_number].da_dz
        );

        //4.  After calculating dL_dz for the output layer (last_layer_number)
        // Propagate to previous layer if it exists.Basically this means if the last layer number is more than 0, it means there is a layer before the output layer since 0 is the last layer.
        if (layer_number > 0) {
            assert(Matrix_Can_Multiply(intermediate_matrices[layer_number-1].dL_da_upper_layer,
                                       intermediate_matrices[layer_number].dL_dz,
                                       intermediate_matrices[layer_number].W_transposed));
            Matrix_Multiply(
                    intermediate_matrices[layer_number-1].dL_da_upper_layer,
                    intermediate_matrices[layer_number].dL_dz,
                    intermediate_matrices[layer_number].W_transposed
            );
        }

        // 5. Calculate gradient of loss with respect to weights (dL/dW)
        // dL/dW = (input)^T * dL/dz

// dL/dW = (input)^T * dL/dz
        if(intermediate_matrices[layer_number].dL_dW.rows()==layer_information[layer_number].weights_matrix.rows() &&
           intermediate_matrices[layer_number].dL_dW.columns()==layer_information[layer_number].weights_matrix.columns()){
            Matrix_Resize(intermediate_matrices[layer_number].dL_dW,layer_information[layer_number].weights_matrix.columns(),
                          layer_information[layer_number].weights_matrix.rows());//this is just to resize the matrix so that it can
            // be multiplied
        }
        assert(Matrix_Can_Multiply(intermediate_matrices[layer_number].dL_dW, intermediate_matrices[layer_number].I_transposed, intermediate_matrices[layer_number].dL_dz));
        Matrix_Multiply(intermediate_matrices[layer_number].dL_dW, intermediate_matrices[layer_number].I_transposed, intermediate_matrices[layer_number].dL_dz);

        Matrix temp=intermediate_matrices[layer_number].dL_dW;
        Matrix_Resize(intermediate_matrices[layer_number].dL_dW,layer_information[layer_number].weights_matrix.rows(),
                      layer_information[layer_number].weights_matrix.columns());
        Matrix_Transpose(intermediate_matrices[layer_number].dL_dW, temp);


        // 5. Calculate gradient of loss with respect to biases (dL/db)
        // dL/db = dL/dz (sum across batch)
        intermediate_matrices[layer_number].dL_db=intermediate_matrices[layer_number].dL_dz;
    }


}

// Update weights and biases using calculated gradients
void Train_Block_by_Backpropagation::UpdateLayerParameters() {
    int size_of_block= neural_block.Get_Block_Size();


    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {
        // Update weights: W = W - learning_rate * dL/dW
        Matrix intermediate_weight_matrix =intermediate_matrices[layer_number].dL_dW ;
        Matrix_Scalar_Multiply(intermediate_weight_matrix,learning_rate);
        assert(Matrix_Can_AddOrSubtract(layer_information[layer_number].weights_matrix,layer_information[layer_number].weights_matrix,intermediate_weight_matrix));
        Matrix_Subtract(layer_information[layer_number].weights_matrix,layer_information[layer_number].weights_matrix,intermediate_weight_matrix);

        // Update biases: b = b - learning_rate * dL/db
        Matrix intermediate_bias_matrix = intermediate_matrices[layer_number].dL_db;
        Matrix_Scalar_Multiply(intermediate_bias_matrix,learning_rate);
        assert(Matrix_Can_AddOrSubtract(layer_information[layer_number].bias_matrix,layer_information[layer_number].bias_matrix,intermediate_bias_matrix));
        Matrix_Subtract(layer_information[layer_number].bias_matrix,layer_information[layer_number].bias_matrix,intermediate_bias_matrix);

    }
    UpdateLayerInformation();
    neural_block.Calculate_Block_Loss();
    cost=neural_block.Get_Loss();


}

void Train_Block_by_Backpropagation::UpdateLayerInformation(){
    int size_of_block= neural_block.Get_Block_Size();

    //starting form the first layer, let us now update the renewed matrices after one step of backprop has been completed. Weight and Bias has already been updated.
    for (int layer_number = 0; layer_number < size_of_block; ++layer_number) {
        if (layer_number > 0) {
            layer_information[layer_number].input_matrix = intermediate_matrices[layer_number-1].post_activation_tensor;  // Initialize the first layer's input
        }
        layer_information[layer_number].pre_activation_tensor=intermediate_matrices[layer_number].pre_activation_tensor;
        layer_information[layer_number].post_activation_tensor=intermediate_matrices[layer_number].post_activation_tensor;


        neural_block.Set_Layers(layer_number).input_matrix=layer_information[layer_number].input_matrix ;
        neural_block.Set_Layers(layer_number).weights_matrix=layer_information[layer_number].weights_matrix ;
        neural_block.Set_Layers(layer_number).bias_matrix=layer_information[layer_number].bias_matrix ;
        neural_block.Set_Layers(layer_number).pre_activation_tensor=layer_information[layer_number].pre_activation_tensor;
        neural_block.Set_Layers(layer_number).post_activation_tensor=layer_information[layer_number].post_activation_tensor;

    }

}

// Add this to the public methods section of Neural_Block class
Matrix Train_Block_by_Backpropagation::Get_Predictions() const {
    // Return post-activation tensor of the last layer
    int last_layer = neural_block.Get_Block_Size() - 1;
    return neural_block.Get_Layers(last_layer).post_activation_tensor;
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




