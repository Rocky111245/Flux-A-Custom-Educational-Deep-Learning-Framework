

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_UTILITY_MIGRATORY_FUNCTIONS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_UTILITY_MIGRATORY_FUNCTIONS_H
#include <fstream>
#include <iostream>
#include <vector>

class Matrix;
class Kernel;
class Tensor;


Matrix MNIST_Single_Targeted_Image_To_Matrix(const std::string& filename, int image_index);
void Print_MNIST_Image_In_Console(const Matrix& mnist_image);
float Matrix_Dot_Product(const Matrix& first, const Matrix& second);
void Matrix_Extract_Patches(std::vector<Matrix> &patches, const Matrix &padded_input_matrix, const Kernel &kernel,int padding_size,int stride_size);
void Matrix_Add_Padding(Matrix &padded_matrix, const Matrix &input_matrix, int padding_size);
Tensor Extract_Patches_Im2rows(const Tensor &padded_input_tensor, const int kernel_size, const int padding_size, const int stride_size=1);
int Padding_Size_Needed(int input_rows, int input_columns,int desired_output_rows,int desired_output_columns, int kernel_size,int stride);
void Tensor_Padding_Symmetric(Tensor &padded_tensor, const Tensor &input_tensor, int padding_size);
#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_UTILITY_MIGRATORY_FUNCTIONS_H
