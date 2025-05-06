/**
 * @file Neural_Blocks.cpp
 * @brief Implementation of the Neural_Block class
 * @author Rakib
 * @date 2025-02-14
 */

#include <iomanip>
#include "Neural_Blocks.h"
#include "Utility_Functions/Utility_Functions.h"

//===========================================================================
// Constructors & Destructor
//===========================================================================

Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : input_matrix(input_matrix), layers(layer_list) {
    // Mark that this block has a valid input matrix
    input_matrix_constructed = true;
}

Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : layers(layer_list) {
    // No special initialization needed for middle blocks
}

Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& output_matrix)
        : layers(layer_list), target_matrix(output_matrix), lossFunction(loss_function) {
    // Mark that this block has valid output and loss function
    target_matrix_constructed = true;
    loss_function_constructed = true;
}

Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& target_matrix)
        : input_matrix(input_matrix), target_matrix(target_matrix),
          layers(layer_list), lossFunction(loss_function) {
    // Mark that this is a complete block with all components
    input_matrix_constructed = true;
    target_matrix_constructed = true;
    loss_function_constructed = true;

    // Initialize all matrices to create a complete network
    Construct_Matrices();
}

//This constructor is necessary to make it work for the web assembly
Neural_Block::Neural_Block(Matrix& input_matrix, const std::vector<Neural_Layer_Skeleton>& layer_vector,
                           LossFunction loss_function, Matrix& target_matrix)
        : input_matrix(input_matrix), target_matrix(target_matrix),
          layers(layer_vector), lossFunction(loss_function) {
    // Mark that this is a complete block with all components
    input_matrix_constructed = true;
    target_matrix_constructed = true;
    loss_function_constructed = true;

    // Initialize all matrices to create a complete network
    Construct_Matrices();
}

Neural_Block::Neural_Block(const Neural_Block& other)
        : input_matrix(other.input_matrix),
          target_matrix(other.target_matrix),
          lossFunction(other.lossFunction),
          loss(other.loss),
          layers(other.layers),
          input_matrix_constructed(other.input_matrix_constructed),
          target_matrix_constructed(other.target_matrix_constructed),
          loss_function_constructed(other.loss_function_constructed) {
    // All members are deep-copied through their respective copy constructors
}

Neural_Block::Neural_Block(Neural_Block&& other) noexcept
        : input_matrix(std::move(other.input_matrix)),
          target_matrix(std::move(other.target_matrix)),
          lossFunction(other.lossFunction),
          loss(other.loss),
          layers(std::move(other.layers)),
          input_matrix_constructed(other.input_matrix_constructed),
          target_matrix_constructed(other.target_matrix_constructed),
          loss_function_constructed(other.loss_function_constructed) {

    // Reset the moved-from object to a valid but empty state
    other.input_matrix_constructed = false;
    other.target_matrix_constructed = false;
    other.loss_function_constructed = false;
    other.loss = 0.0f;
}

Neural_Block& Neural_Block::operator=(const Neural_Block& other) {
    if (this != &other) { // Prevent self-assignment
        // Perform deep copy of all members
        input_matrix = other.input_matrix;
        target_matrix = other.target_matrix;
        lossFunction = other.lossFunction;
        loss = other.loss;
        layers = other.layers;
        input_matrix_constructed = other.input_matrix_constructed;
        target_matrix_constructed = other.target_matrix_constructed;
        loss_function_constructed = other.loss_function_constructed;
    }
    return *this;
}

Neural_Block& Neural_Block::operator=(Neural_Block&& other) noexcept {
    if (this != &other) { // Prevent self-assignment
        // Move resources efficiently
        input_matrix = std::move(other.input_matrix);
        target_matrix = std::move(other.target_matrix);
        lossFunction = other.lossFunction;
        loss = other.loss;
        layers = std::move(other.layers);
        input_matrix_constructed = other.input_matrix_constructed;
        target_matrix_constructed = other.target_matrix_constructed;
        loss_function_constructed = other.loss_function_constructed;

        // Reset the moved-from object to a valid but empty state
        other.input_matrix_constructed = false;
        other.target_matrix_constructed = false;
        other.loss_function_constructed = false;
        other.loss = 0.0f;
    }
    return *this;
}

Neural_Block::~Neural_Block() {
    // Most resources are managed by RAII through matrices and vectors
    // No manual cleanup needed
}

//===========================================================================
// Public Methods
//===========================================================================

void Neural_Block::Forward_Pass_With_Activation() {
    size_t size_of_layer = layers.size();

    // Forward pass through each layer sequentially
    for (size_t i = 0; i < size_of_layer; i++) {
        // Compute pre-activation (z = xW + b)
        Compute_PreActivation_Matrix(layers[i].input_matrix, layers[i].weights_matrix,
                                     layers[i].bias_matrix, layers[i].pre_activation_tensor);

        // Apply activation function (a = g(z))
        Compute_PostActivation_Matrix(layers[i].pre_activation_tensor,
                                      layers[i].post_activation_tensor, layers[i].activationType);

        // Set the output of this layer as input to the next layer
        if (i < size_of_layer - 1) {
            layers[i+1].input_matrix = layers[i].post_activation_tensor;
        }
    }

    // Calculate network loss after completing the forward pass
    Calculate_Block_Loss();
}

Neural_Block& Neural_Block::Connect_With(Neural_Block& block2) {
    // Verify that connection is valid - neither block should be complete
    if (input_matrix_constructed && target_matrix_constructed && loss_function_constructed) {
        std::cerr << "Error: Cannot connect blocks if they are already complete." << std::endl;
        exit(1);
    }

    // The output block should be provided as argument, not be the caller
    if (this->target_matrix_constructed && this->loss_function_constructed) {
        std::cerr << "Error: CANNOT connect. Output block should be the argument, not the caller." << std::endl;
        exit(1);
    }

    // Append the layers from block2 to this block
    layers.insert(layers.end(), block2.layers.begin(), block2.layers.end());

    // If combining these blocks creates a complete network, initialize all matrices
    if (input_matrix_constructed && block2.target_matrix_constructed && block2.loss_function_constructed) {
        Construct_Matrices();
    }

    return *this;
}

//===========================================================================
// Getters
//===========================================================================

Neural_Layer_Skeleton& Neural_Block::Set_Block_Layers(int layer_number) {
    return layers[layer_number];
}

bool Neural_Block::Get_Block_Status() const {
    // A block is complete when it has input, target, and loss function
    return (input_matrix_constructed && target_matrix_constructed && loss_function_constructed);
}

LossFunction Neural_Block::Get_Block_Loss_Type() const {
    return lossFunction;
}

int Neural_Block::Get_Block_Size() const {
    return layers.size();
}

float Neural_Block::Get_Block_Loss() const {
    return loss;
}

const Matrix& Neural_Block::Get_Block_Target_Matrix() const {
    return target_matrix;
}

Matrix Neural_Block::Get_Block_Input_Matrix() const {
    return input_matrix;
}

Matrix& Neural_Block::Get_Block_Weights_Matrix(int layer_number) {
    return layers[layer_number].weights_matrix;
}

Matrix& Neural_Block::Get_Block_Bias_Matrix(int layer_number) {
    return layers[layer_number].bias_matrix;
}

Matrix& Neural_Block::Get_Block_Pre_Activation_Matrix(int layer_number) {
    return layers[layer_number].pre_activation_tensor;
}

Matrix& Neural_Block::Get_Block_Post_Activation_Matrix(int layer_number) {
    return layers[layer_number].post_activation_tensor;
}

//===========================================================================
// Setters
//===========================================================================

Neural_Layer_Skeleton& Neural_Block::Set_Layers(int layer_number) {
    return layers[layer_number];
}

//===========================================================================
// Private Helper Methods
//===========================================================================

// Utility function to print a matrix (for debugging)
void Matrix_Print(Matrix& matrix) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.columns(); j++) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void Neural_Block::Construct_Matrices() {
    // Initialize matrices for each layer with proper dimensions
    for (size_t i = 0; i < layers.size(); i++) {
        // Set up input connections
        if (i == 0) {
            // First layer gets the block's input
            layers[i].input_matrix = input_matrix;
        } else {
            // Other layers connect to previous layer's output
            layers[i].input_matrix = layers[i - 1].post_activation_tensor;
        }

        // Initialize weights with Xavier/Glorot initialization for better convergence
        layers[i].weights_matrix = Matrix(layers[i].get_neuron_count(), layers[i].input_matrix.columns(), 0.0f);
        Matrix_Xavier_Uniform(layers[i].weights_matrix);

        // Initialize bias, pre-activation and post-activation matrices
        layers[i].bias_matrix = Matrix(1, layers[i].get_neuron_count(), 0.0f);
        layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
        layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
    }
}

void Neural_Block::Compute_PreActivation_Matrix(Matrix& input_matrix_internal,
                                                Matrix& weights_matrix_internal,
                                                Matrix& bias_matrix_internal,
                                                Matrix& pre_activation_tensor_internal) {
    // Create transposed weights for efficient matrix multiplication
    Matrix transposed_weights(weights_matrix_internal.columns(), weights_matrix_internal.rows(), 0.0f);
    Matrix_Transpose(transposed_weights, weights_matrix_internal);

    // Ensure matrices have compatible dimensions for multiplication
    assert(Matrix_Can_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights));

    // Compute z = x * W^T (matrix multiplication)
    Matrix_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights);

    // Apply bias: z = x * W^T + b (element-wise addition)
    // First broadcast bias to match pre-activation dimensions
    Matrix temp(pre_activation_tensor_internal.rows(), pre_activation_tensor_internal.columns(), 0.0f);
    Matrix_Broadcast(temp, bias_matrix_internal, pre_activation_tensor_internal.rows(), pre_activation_tensor_internal.columns());
    Matrix_Add(pre_activation_tensor_internal, pre_activation_tensor_internal, temp);
}

void Neural_Block::Calculate_Block_Loss() {
    // Verify the block is complete before calculating loss
    assert(Get_Block_Status());

    // Get the output of the last layer
    int last_layer = layers.size() - 1;

    // Calculate loss between network output and target
    loss = Calculate_Loss(layers[last_layer].post_activation_tensor, target_matrix, lossFunction);
}

void Neural_Block::Compute_PostActivation_Matrix(Matrix& pre_activation_tensor_internal,
                                                 Matrix& post_activation_tensor_internal,
                                                 ActivationType activation_function_internal) {
    // Apply the chosen activation function to the pre-activation values
    Apply_Activation_Function_To_Matrix(post_activation_tensor_internal, pre_activation_tensor_internal, activation_function_internal);
}

void Neural_Block::Apply_Activation_Function_To_Matrix(Matrix& result, const Matrix& input, ActivationType activation_type) {
    // Ensure matrices have compatible dimensions
    if (result.rows() != input.rows() || result.columns() != input.columns()) {
        throw std::invalid_argument("Input and result matrices must have matching dimensions.");
    }

    // Apply the appropriate activation function element-wise
    switch (activation_type) {
        case ActivationType::RELU: {
            // ReLU: f(x) = max(0, x)
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value > 0 ? value : 0;
                }
            }
            break;
        }
        case ActivationType::SIGMOID: {
            // Sigmoid: f(x) = 1/(1+e^(-x))
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = 1.0f / (1.0f + std::exp(-value));
                }
            }
            break;
        }
        case ActivationType::TANH: {
            // Hyperbolic tangent: f(x) = tanh(x)
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = std::tanh(value);
                }
            }
            break;
        }
        case ActivationType::LEAKY_RELU: {
            // Leaky ReLU: f(x) = x if x > 0, else 0.01*x
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value > 0 ? value : 0.01f * value;
                }
            }
            break;
        }
        case ActivationType::SWISH: {
            // Swish: f(x) = x * sigmoid(x)
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value * (1.0f / (1.0f + std::exp(-value)));
                }
            }
            break;
        }
        case ActivationType::LINEAR: {
            // Linear: f(x) = x
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    result(i, j) = input(i, j);
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Invalid activation function.");
    }
}