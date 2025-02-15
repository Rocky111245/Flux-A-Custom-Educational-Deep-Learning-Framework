//
// Created by rakib on 14/2/2025.
//

#include "Neural_Layers/Neural_Layer_Skeleton.h"
#include "Neural_Blocks.h"
#include "Neural Network Framework.h"
#include <MatrixLibrary.h>
#include "Loss Functions/Loss_Functions.h"


//this class defines a neural layer which comprises of many different neurones. Neurones accept data from previous layer and provides an output tensor.


float Neural_Layer::Sigmoid_Function(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float Neural_Layer::ReLU(float x) {
    return std::max(0.0f, x);
}

float Neural_Layer::LeakyReLU(float x) {
    float alpha = 0.01;  // Example value, can be changed to the desired slope for negative inputs
    return (x > 0) ? x : alpha * x;
}


float Neural_Layer::ELU(float x, float alpha) {
    return (x > 0) ? x : alpha * (std::exp(x) - 1);
}

float Neural_Layer::Swish(float x) {
    return x * (1.0f / (1.0f + std::exp(-x)));
}

float Neural_Layer::Tanh(float x) {
    return std::tanh(x);
}

float Neural_Layer::Linear_Activation(float x) {
    return x;
}


class Neural_Block{
public:
    Neural_Block()= default;

    //this is a basic network in which the user does not initialize their own weights and relies on the framework to do it for them
    // Constructor that accepts an initializer list
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list)
            : input_matrix(input_matrix),
             layers(layer_list) {  // Automatically converts the list to a vector
        //this initializes the matrices to correct dimensions so that calculations can be done
        Construct_Matrices();
    }
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list, LossFunction loss_function,Matrix& output_matrix)
            : input_matrix(input_matrix),
              output_matrix(output_matrix),
              layers(layer_list) {  // Automatically converts the list to a vector
        //this initializes the matrices to correct dimensions so that calculations can be done
        Construct_Matrices();
    }

    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list)
            :layers(layer_list) {  // Automatically converts the list to a vector
        //this initializes the matrices to correct dimensions so that calculations can be done
        Construct_Matrices();
    }
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list, LossFunction loss_function,Matrix& output_matrix)
            :layers(layer_list),
            output_matrix(output_matrix)


            {  // Automatically converts the list to a vector
        //this initializes the matrices to correct dimensions so that calculations can be done
        Construct_Matrices();
    }


    //this is a basic network in which the user initializes their own weights
    Neural_Block(Matrix &input_matrix, Matrix user_weights_matrix){

    }

    Neural_Block(Matrix &input_matrix, ActivationType activation_function){

    }


    //This matrix computes the simple forward pass calculations
    void Forward_Pass_With_Activation(){
        size_t size_of_layer=layers.size();
        for(int i=0;i<size_of_layer;i++){
            Compute_PreActivation_Matrix(layers[i].input_matrix,layers[i].weights_matrix,layers[i].bias_matrix,layers[i].pre_activation_tensor);
            Compute_PostActivation_Matrix(layers[i].pre_activation_tensor,layers[i].post_activation_tensor,layers[i].activationType);
        }
    }

    //this function connects block to block
    void Connect_With( Neural_Block& block2){
        int last_layer=layers.size()-1;
        block2.layers[0].input_matrix=layers[last_layer].post_activation_tensor;
        if(block2.layers[0].input_matrix.columns()!=block2.layers[0].weights_matrix.rows()){
            std::cout<<"The number of columns in the input matrix does not match the number of rows in the weights matrix"<<std::endl;
            return;
        }
    }

    void Calculate_Block_Loss(LossFunction loss_function){
        int last_layer=layers.size()-1;
        loss=Calculate_Loss(layers[last_layer].post_activation_tensor, output_matrix, loss_function);
    }






private:
    Matrix input_matrix;//initialized to the input matrix from the user
    Matrix output_matrix;
    float loss;

    std::vector<Neural_Layer_Skeleton> layers;

    std::pair<int,ActivationType> neural_layer_skeleton_info;



    //this updates the pre-activation matrix
    void Compute_PreActivation_Matrix(Matrix &input_matrix_internal, Matrix &weights_matrix_internal, Matrix &bias_matrix_internal, Matrix &pre_activation_tensor_internal){
        Matrix_Transpose(weights_matrix_internal,weights_matrix_internal);
        Matrix_Multiply(pre_activation_tensor_internal, input_matrix_internal, weights_matrix_internal);
        Matrix_Add(pre_activation_tensor_internal,pre_activation_tensor_internal,bias_matrix_internal);
    }
    //this updates the post-activation matrix
    void Compute_PostActivation_Matrix(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,ActivationType activation_function_internal){
        Apply_Activation(pre_activation_tensor_internal,post_activation_tensor_internal,activation_function_internal);
    }

    void Apply_Activation(Matrix &pre_activation_tensor_internal,Matrix &post_activation_tensor_internal,ActivationType activation_function){
        switch (activation_function) {
            case ActivationType::RELU:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::ReLU(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            case ActivationType::SIGMOID:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::Sigmoid_Function(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            case ActivationType::TANH:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::Tanh(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            case ActivationType::LEAKY_RELU:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::LeakyReLU(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            case ActivationType::SWISH:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::Swish(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            case ActivationType::LINEAR:
                for(int i=0;i<pre_activation_tensor_internal.rows();i++){
                    for(int j=0;j<pre_activation_tensor_internal.columns();j++){
                        post_activation_tensor_internal(i,j)=Neural_Layer::Linear_Activation(pre_activation_tensor_internal(i,j));
                    }
                }
                break;
            default:
                throw std::invalid_argument("Invalid loss function type for this overload.");
        }

    }





//after the layers have been defined, this function constructs the overall skeleton of the network (initializes the matrices)
    void Construct_Matrices() {

        for (int i = 0; i < layers.size()-1; i++) {
            if (i == 0) {
                layers[i].input_matrix = input_matrix; //only the input is initialized to the input matrix
                layers[i].weights_matrix = Matrix(layers[i + 1].get_neuron_count(),input_matrix.columns(),  0.0f);
                layers[i].bias_matrix = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(),0.0f);
                layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
                layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
            } else {
                layers[i].input_matrix = Matrix(layers[i-1].post_activation_tensor.rows(),layers[i-1].post_activation_tensor.columns(),0.0f);
                layers[i].weights_matrix = Matrix(layers[i + 1].get_neuron_count(),layers[i].input_matrix.columns(),0.0f);
                layers[i].bias_matrix = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
                layers[i].pre_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
                layers[i].post_activation_tensor = Matrix(layers[i].input_matrix.rows(), layers[i].weights_matrix.columns(), 0.0f);
            }
        }


    }
};




int main(){
    Matrix input(32, 784, 0.0f);  // Example: MNIST dataset with batch size 32
    Matrix output(32, 10, 0.0f);

// Define the network architecture
Neural_Layer_Skeleton layer1(784, ActivationType::LINEAR);  // Input layer
Neural_Layer_Skeleton layer2(256, ActivationType::RELU);    // Hidden layer 1
Neural_Layer_Skeleton layer3(128, ActivationType::RELU);    // Hidden layer 2
Neural_Layer_Skeleton layer4(10, ActivationType::SIGMOID);  // Output layer
Neural_Layer_Skeleton layer5(10, ActivationType::SIGMOID);  // Output layer



    Neural_Block block1(input, {
        layer1,
        layer2,
        layer3,
        layer4
    },LossFunction::CROSS_ENTROPY_LOSS,output);

    Neural_Block block2({
            layer5
    });



block1.Forward_Pass_With_Activation();

block1.Connect_With(block2);

}







