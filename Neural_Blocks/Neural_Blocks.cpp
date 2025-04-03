//
// Created by rakib on 14/2/2025.
//

#include <iomanip>
#include "Neural_Blocks.h"

//===========================================================================
// Constructors & Destructor
//===========================================================================

// Constructor that accepts an input matrix and a list of layers, it does not have an output matrix.
// First block if blocks are to be connected.
Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : input_matrix(input_matrix), layers(layer_list) {
    input_matrix_constructed = true;
}

// Constructor for blocks without predefined input. Blocks in middle
Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : layers(layer_list) {
}

// Constructor for blocks with loss function and output matrix. The last block
Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& output_matrix)
        : layers(layer_list), target_matrix(output_matrix), lossFunction(loss_function) {
    target_matrix_constructed = true;
    loss_function_constructed = true;
}

// Constructor that includes loss function and output matrix. This is one full block
Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& target_matrix)
        : input_matrix(input_matrix), target_matrix(target_matrix),
          layers(layer_list), lossFunction(loss_function) {
    input_matrix_constructed = true;
    target_matrix_constructed = true;
    loss_function_constructed = true;
    Construct_Matrices();
}

// Copy constructor
Neural_Block::Neural_Block(const Neural_Block& other)
        : input_matrix(other.input_matrix),
          target_matrix(other.target_matrix),
          lossFunction(other.lossFunction),
          loss(other.loss),
          layers(other.layers),
          input_matrix_constructed(other.input_matrix_constructed),
          target_matrix_constructed(other.target_matrix_constructed),
          loss_function_constructed(other.loss_function_constructed) {
}

// Move constructor
Neural_Block::Neural_Block(Neural_Block&& other) noexcept
        : input_matrix(std::move(other.input_matrix)),
          target_matrix(std::move(other.target_matrix)),
          lossFunction(other.lossFunction),
          loss(other.loss),
          layers(std::move(other.layers)),
          input_matrix_constructed(other.input_matrix_constructed),
          target_matrix_constructed(other.target_matrix_constructed),
          loss_function_constructed(other.loss_function_constructed) {

    // Reset the moved-from object's state
    other.input_matrix_constructed = false;
    other.target_matrix_constructed = false;
    other.loss_function_constructed = false;
    other.loss = 0.0f;
}

// Copy assignment operator
Neural_Block& Neural_Block::operator=(const Neural_Block& other) {
    if (this != &other) { // Prevent self-assignment
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

// Move assignment operator
Neural_Block& Neural_Block::operator=(Neural_Block&& other) noexcept {
    if (this != &other) { // Prevent self-assignment
        input_matrix = std::move(other.input_matrix);
        target_matrix = std::move(other.target_matrix);
        lossFunction = other.lossFunction;
        loss = other.loss;
        layers = std::move(other.layers);
        input_matrix_constructed = other.input_matrix_constructed;
        target_matrix_constructed = other.target_matrix_constructed;
        loss_function_constructed = other.loss_function_constructed;

        // Reset the moved-from object's state
        other.input_matrix_constructed = false;
        other.target_matrix_constructed = false;
        other.loss_function_constructed = false;
        other.loss = 0.0f;
    }
    return *this;
}

// Destructor
Neural_Block::~Neural_Block() {
    // Most resources are managed by RAII through matrices and vectors
    // No manual cleanup needed
}

//===========================================================================
// Public Methods
//===========================================================================

// Perform forward pass with activation
void Neural_Block::Forward_Pass_With_Activation() {
    size_t size_of_layer = layers.size();

    // Forward pass through each layer
    for (size_t i = 0; i < size_of_layer; i++) {
        // Compute this layer's activations
        Compute_PreActivation_Matrix(layers[i].input_matrix, layers[i].weights_matrix,
                                     layers[i].bias_matrix, layers[i].pre_activation_tensor);
        Compute_PostActivation_Matrix(layers[i].pre_activation_tensor,
                                      layers[i].post_activation_tensor, layers[i].activationType);

        // Update input for next layer
        if (i < size_of_layer - 1) {
            layers[i+1].input_matrix = layers[i].post_activation_tensor;
        }
    }

    // Calculate loss after forward pass is complete
    Calculate_Block_Loss();
}

// Connect one block with another. It makes one new big block
Neural_Block& Neural_Block::Connect_With(Neural_Block& block2) {
    // Connection is only possible if it is not a full block. If it is a full block, we cannot connect.
    if (input_matrix_constructed && target_matrix_constructed && loss_function_constructed) {
        std::cerr << "Error: Cannot connect blocks if they are already complete." << std::endl;
        exit(1);
    }

    if (this->target_matrix_constructed && this->loss_function_constructed) {
        std::cerr << "Error: CANNOT connect. Output block should be the argument, not the caller." << std::endl;
        exit(1);
    }

    layers.insert(layers.end(), block2.layers.begin(), block2.layers.end());

    // If the block has all of these 3, this means that the connection has completed.
    if (input_matrix_constructed && block2.target_matrix_constructed && block2.loss_function_constructed) {
        Construct_Matrices();
    }
    return *this;
}




//===========================================================================
// Getters
//===========================================================================

const Neural_Layer_Skeleton& Neural_Block::Get_Block_Layers(int layer_number) const {
    return layers[layer_number];
}

bool Neural_Block::Get_Block_Status() const {
    if (input_matrix_constructed && target_matrix_constructed && loss_function_constructed) {
        return true;
    }
    return false;
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

//Used to update the matrices after a layer has been modified in backpropagation
Neural_Layer_Skeleton& Neural_Block::Set_Layers(int layer_number) {
    return layers[layer_number];
}

//===========================================================================
// Private Helper Methods
//===========================================================================
void Matrix_Print(Matrix& matrix) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.columns(); j++) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
// Construct matrices for each layer with correct dimensions
void Neural_Block::Construct_Matrices() {
    //set the output to the last layer

    for (size_t i = 0; i < layers.size(); i++) {
        if (i == 0) {
            layers[i].input_matrix = input_matrix;  // Initialize the first layer's input
        } else {
            layers[i].input_matrix = layers[i - 1].post_activation_tensor;  // Connect layers
        }

        // All matrices initialized to zero
        layers[i].weights_matrix = Matrix(layers[i].get_neuron_count(), layers[i].input_matrix.columns(), 0.0f);
        Matrix_Xavier_Uniform(layers[i].weights_matrix);

        layers[i].bias_matrix = Matrix(1, layers[i].get_neuron_count(), 0.0f);
        layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
        layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
    }
}



// Compute pre-activation matrix
void Neural_Block::Compute_PreActivation_Matrix(Matrix& input_matrix_internal,
                                                Matrix& weights_matrix_internal,
                                                Matrix& bias_matrix_internal,
                                                Matrix& pre_activation_tensor_internal) {
    Matrix transposed_weights(weights_matrix_internal.columns(), weights_matrix_internal.rows(), 0.0f);
    Matrix_Transpose(transposed_weights, weights_matrix_internal);

    assert(Matrix_Can_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights));
    //Debug_Matrix(input_matrix_internal, "input_matrix_internal");
    //Debug_Matrix(weights_matrix_internal, "weights_matrix_internal");
    //Debug_Matrix(transposed_weights, "transposed_weights");
    //Debug_Matrix(pre_activation_tensor_internal, "pre_activation_tensor_internal");
    Matrix_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights);
    //Debug_Matrix(pre_activation_tensor_internal, "After calculation,before bias addition: pre_activation_tensor_internal");

    //Apply bias
    Matrix temp(pre_activation_tensor_internal.rows(), pre_activation_tensor_internal.columns(), 0.0f);
    Matrix_Broadcast(temp, bias_matrix_internal, pre_activation_tensor_internal.rows(), pre_activation_tensor_internal.columns());
    //Debug_Matrix(temp, "Bias Broadcasted Matrix");
    Matrix_Add(pre_activation_tensor_internal, pre_activation_tensor_internal, temp);
    //Debug_Matrix(pre_activation_tensor_internal, "After calculation,after bias addition: pre_activation_tensor_internal");

}

// Compute the loss for a block
void Neural_Block::Calculate_Block_Loss() {
    assert(Get_Block_Status());
    int last_layer = layers.size() - 1;
    loss = Calculate_Loss(layers[last_layer].post_activation_tensor, target_matrix, lossFunction);
}

// Compute post-activation matrix
void Neural_Block::Compute_PostActivation_Matrix(Matrix& pre_activation_tensor_internal,Matrix& post_activation_tensor_internal,ActivationType activation_function_internal) {
    Apply_Activation_Function_To_Matrix( post_activation_tensor_internal,pre_activation_tensor_internal, activation_function_internal);
}

// Apply activation function to an entire matrix with improved efficiency
void Neural_Block::Apply_Activation_Function_To_Matrix(Matrix& result, const Matrix& input, ActivationType activation_type) {
    // Validate matrix dimensions
    if (result.rows() != input.rows() || result.columns() != input.columns()) {
        throw std::invalid_argument("Input and result matrices must have matching dimensions.");
    }

    // Determine activation function once before processing elements
    switch (activation_type) {
        case ActivationType::RELU: {
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value > 0 ? value : 0;
                }
            }
            break;
        }
        case ActivationType::SIGMOID: {
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = 1.0f / (1.0f + std::exp(-value));
                }
            }
            break;
        }
        case ActivationType::TANH: {
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = std::tanh(value);
                }
            }
            break;
        }
        case ActivationType::LEAKY_RELU: {
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value > 0 ? value : 0.01f * value;
                }
            }
            break;
        }
        case ActivationType::SWISH: {
            for (int i = 0; i < input.rows(); i++) {
                for (int j = 0; j < input.columns(); j++) {
                    float value = input(i, j);
                    result(i, j) = value * (1.0f / (1.0f + std::exp(-value)));
                }
            }
            break;
        }
        case ActivationType::LINEAR: {
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