#include <iostream>
#include <MatrixLibrary.h>
#include "Matrix_Migration_Functions/Matrix_Migration_Functions.h"


int main() {
    try {
        // Load the first image from the MNIST training set
        Matrix digit = MNIST_Single_Targeted_Image_To_Matrix("D:\\Github Files\\ Discriminative Dense Neural Network Framework\\FLUX\\Convolutional_Neural_Networks\\IDX_GrayScale_Images\\train-images.idx3-ubyte", 0);

        // Print the image
        std::cout << "MNIST Digit Visualization:" << std::endl;
        Print_MNIST_Image_In_Console(digit);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}