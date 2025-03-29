//
// Created by rakib on 14/2/2025.
//

#include <iomanip>
#include "Neural_Blocks.h"


// Constructor that accepts an input matrix and a list of layers, it does not have an output matrix. First block if blocks are to be connected.
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
        : layers(layer_list), output_matrix(output_matrix),lossFunction(loss_function)  {
    output_matrix_constructed = true;
    loss_function_constructed = true;
}


// Constructor that includes loss function and output matrix. This is one full block
Neural_Block::Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                           LossFunction loss_function, Matrix& output_matrix)
        : input_matrix(input_matrix), output_matrix(output_matrix), layers(layer_list),lossFunction(loss_function) {
    input_matrix_constructed = true;
    output_matrix_constructed = true;
    loss_function_constructed = true;
    Construct_Matrices();

}


const Neural_Layer_Skeleton &Neural_Block::Get_Layers(int layer_number) const{
    return layers[layer_number];
}

Neural_Layer_Skeleton &Neural_Block::Set_Layers(int layer_number) {
    return layers[layer_number];
}

Matrix& Neural_Block::Get_Weights_Matrix(int layer_number) {
        return layers[layer_number].weights_matrix;
}

Matrix& Neural_Block::Get_Bias_Matrix(int layer_number) {
    return layers[layer_number].bias_matrix;
}

bool Neural_Block::Get_Block_Status() const{
    if(input_matrix_constructed && output_matrix_constructed && loss_function_constructed){
        return true;
    }
    return false;
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

// Connect one block with another. It makes one new big block
Neural_Block& Neural_Block::Connect_With(Neural_Block& block2) {
    //connection is only possible if it is not a full block. If it is a full block, we cannot connect.
    if(input_matrix_constructed && output_matrix_constructed && loss_function_constructed){
        std::cerr << "Error: Cannot connect blocks if they are already complete." << std::endl;
        exit(1);
    }

    if(this->output_matrix_constructed && this->loss_function_constructed){
        std::cerr << "Error: CANNOT connect. Output block should be the argument,not the caller." << std::endl;
        exit(1);
    }

    layers.insert(layers.end(), block2.layers.begin(), block2.layers.end());
    //if the block has all of these 3,this means that the connection has completed.
    if(input_matrix_constructed && block2.output_matrix_constructed && block2.loss_function_constructed){
        Construct_Matrices();
    }
    return *this;

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

//This returns an array[row number, column number] of the input to the layer in the block

std::pair<int,int> Neural_Block::Get_Layer_Input_Information(int layer_number) const {
    assert(layer_number<layers.size());

    if(layer_number==0){
        return {input_matrix.rows(),input_matrix.columns()};
    }

    return {layers[layer_number].input_matrix.rows(),layers[layer_number].input_matrix.columns()};
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

        layers[i].bias_matrix = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
        layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);
        layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].get_neuron_count(), 0.0f);

    }
}

// Compute pre-activation matrix
void Neural_Block::Compute_PreActivation_Matrix(Matrix &input_matrix_internal, Matrix &weights_matrix_internal,Matrix &bias_matrix_internal, Matrix &pre_activation_tensor_internal) {

    Matrix transposed_weights(weights_matrix_internal.columns(), weights_matrix_internal.rows(),0.0f);
    Matrix_Transpose(transposed_weights,weights_matrix_internal);
    assert(Matrix_Can_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights));
    Matrix_Multiply(pre_activation_tensor_internal, input_matrix_internal, transposed_weights);
    assert(Matrix_Can_AddOrSubtract(pre_activation_tensor_internal, pre_activation_tensor_internal, bias_matrix_internal));
    Matrix_Add(pre_activation_tensor_internal, pre_activation_tensor_internal, bias_matrix_internal);

}

// Compute post-activation matrix
void Neural_Block::Compute_PostActivation_Matrix(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,
                                                 ActivationType activation_function_internal) {
    Apply_Activation(pre_activation_tensor_internal, post_activation_tensor_internal, activation_function_internal);
}

// Updated Neural_Block::Apply_Activation method
void Neural_Block::Apply_Activation(Matrix &pre_activation_tensor_internal,
                                    Matrix &post_activation_tensor_internal,
                                    ActivationType activation_function) {
    for (int i = 0; i < pre_activation_tensor_internal.rows(); i++) {
        for (int j = 0; j < pre_activation_tensor_internal.columns(); j++) {
            post_activation_tensor_internal(i, j) = ApplyActivationFunction(
                    pre_activation_tensor_internal(i, j),
                    activation_function
            );
        }
    }
}

// Apply activation function to a single value
float Neural_Block::ApplyActivationFunction(float value, ActivationType activation_type) {
    switch (activation_type) {
        case ActivationType::RELU:
            return value > 0 ? value : 0;
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-value));
        case ActivationType::TANH:
            return std::tanh(value);
        case ActivationType::LEAKY_RELU:
            return value > 0 ? value : 0.01f * value;
        case ActivationType::SWISH:
            return value * (1.0f / (1.0f + std::exp(-value)));
        case ActivationType::LINEAR:
            return value;
        default:
            throw std::invalid_argument("Invalid activation function.");
    }
}






