//
// Created by rakib on 11/2/2025.
//

#include "Forward_Pass.h"
#include <MatrixLibrary.h>
#include <Neural Network Framework.h>




//Users can actually define the layers first.
//Users can choose to connect layers together and after this, they may wish to do a forward pass on this layer.

//performs only a single forward pass for modular approach
void Connect_Layers(Matrix &input_vector, Matrix &weights_vector, Matrix &bias_vector, Matrix &output_vector){






}





//performs only a single forward pass for modular approach
void Forward_Pass(Matrix &input_vector, Matrix &weights_vector, Matrix &bias_vector, Matrix &output_vector){

    Matrix weights_vector_transpose;
    //this performs a simple transpose over the weights vector
    Matrix_Transpose(weights_vector_transpose,weights_vector);

    Matrix_Multiply(output_vector,input_vector,weights_vector_transpose);





}