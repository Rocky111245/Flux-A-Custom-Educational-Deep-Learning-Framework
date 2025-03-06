//
// Created by rakib on 14/2/2025.
//

#include "Neural_Blocks.h"


// Constructor that accepts an input matrix and a list of layers, it does not have an output matrix. First block if blocks are to be connected.
Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : input_matrix(input_matrix), layers(layer_list) {
    Construct_Matrices();
}

// Constructor for blocks without predefined input. Blocks in middle
Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list)
        : layers(layer_list) {
    Construct_Matrices();
}

// Constructor for blocks with loss function and output matrix. The last block
Neural_Block::Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& output_matrix)
        : layers(layer_list), output_matrix(output_matrix),lossFunction(loss_function)  {
    Construct_Matrices();
}


// Constructor that includes loss function and output matrix. This is one full block
Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& output_matrix)
        : input_matrix(input_matrix), output_matrix(output_matrix), layers(layer_list),lossFunction(loss_function) {
    Construct_Matrices();
}


Neural_Layer_Skeleton Neural_Block::Get_Layers(int layer_number) const{
    return layers[layer_number];
}


// Perform forward pass with activation
void Neural_Block::Forward_Pass_With_Activation() {
    size_t size_of_layer = layers.size();
    for (size_t i = 0; i < size_of_layer; i++) {
        Compute_PreActivation_Matrix(layers[i].input_matrix, layers[i].weights_matrix,
                                     layers[i].bias_matrix, layers[i].pre_activation_tensor);
        Compute_PostActivation_Matrix(layers[i].pre_activation_tensor,
                                      layers[i].post_activation_tensor, layers[i].activationType);
    }
}

// Connect one block with another
void Neural_Block::Connect_With(Neural_Block& block2) {
    int last_layer = layers.size() - 1;
    block2.layers[0].input_matrix = layers[last_layer].post_activation_tensor;

    if (block2.layers[0].input_matrix.columns() != block2.layers[0].weights_matrix.rows()) {
        std::cout << "Error: Input matrix columns do not match weight matrix rows in connected block." << std::endl;
    }
}

LossFunction Neural_Block::Get_Block_Loss_Type() const{
    return lossFunction;
}

int Neural_Block::Get_Block_Size() const{
    return layers.size();
}

float Neural_Block::Get_Loss() const{
    return loss;
}

Matrix Neural_Block::Get_Output_Matrix() const{
    return output_matrix;
}

// Compute the loss for a block
void Neural_Block::Calculate_Block_Loss() {
    int last_layer = layers.size() - 1;
    loss = Calculate_Loss(layers[last_layer].post_activation_tensor, output_matrix,lossFunction);
}



// Construct matrices for each layer
void Neural_Block::Construct_Matrices() {
    for (size_t i = 0; i < layers.size(); i++) {
        if (i == 0) {
            layers[i].input_matrix = input_matrix;  // Initialize the first layer's input
        } else {
            layers[i].input_matrix = layers[i - 1].post_activation_tensor;  // Connect layers
        }

        //all matrices initialized to zero
        layers[i].weights_matrix = Matrix(layers[i].get_neuron_count(), layers[i].input_matrix.columns(), 0.0f);
        Matrix_Xavier_Uniform(layers[i].weights_matrix);
        layers[i].bias_matrix = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
        layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
        layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
    }
}

// Compute pre-activation matrix
void Neural_Block::Compute_PreActivation_Matrix(Matrix &input_matrix_internal, Matrix &weights_matrix_internal,
                                                Matrix &bias_matrix_internal, Matrix &pre_activation_tensor_internal) {
    Matrix_Transpose(weights_matrix_internal, weights_matrix_internal);
    Matrix_Multiply(pre_activation_tensor_internal, input_matrix_internal, weights_matrix_internal);
    Matrix_Add(pre_activation_tensor_internal, pre_activation_tensor_internal, bias_matrix_internal);
}

// Compute post-activation matrix
void Neural_Block::Compute_PostActivation_Matrix(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,
                                                 ActivationType activation_function_internal) {
    Apply_Activation(pre_activation_tensor_internal, post_activation_tensor_internal, activation_function_internal);
}

// Apply activation function
void Neural_Block::Apply_Activation(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,
                                    ActivationType activation_function) {
    for (int i = 0; i < pre_activation_tensor_internal.rows(); i++) {
        for (int j = 0; j < pre_activation_tensor_internal.columns(); j++) {
            switch (activation_function) {
                case ActivationType::RELU:
                    post_activation_tensor_internal(i, j) = Neural_Layer::ReLU(pre_activation_tensor_internal(i, j));
                    break;
                case ActivationType::SIGMOID:
                    post_activation_tensor_internal(i, j) = Neural_Layer::Sigmoid_Function(pre_activation_tensor_internal(i, j));
                    break;
                case ActivationType::TANH:
                    post_activation_tensor_internal(i, j) = Neural_Layer::Tanh(pre_activation_tensor_internal(i, j));
                    break;
                case ActivationType::LEAKY_RELU:
                    post_activation_tensor_internal(i, j) = Neural_Layer::LeakyReLU(pre_activation_tensor_internal(i, j));
                    break;
                case ActivationType::SWISH:
                    post_activation_tensor_internal(i, j) = Neural_Layer::Swish(pre_activation_tensor_internal(i, j));
                    break;
                case ActivationType::LINEAR:
                    post_activation_tensor_internal(i, j) = Neural_Layer::Linear_Activation(pre_activation_tensor_internal(i, j));
                    break;
                default:
                    throw std::invalid_argument("Invalid activation function.");
            }
        }
    }
}
