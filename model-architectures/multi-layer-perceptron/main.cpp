//
// Created by Rocky170 on 10/28/2025.
//


#include <iostream>
#include <iomanip>
#include "tensor-library/TensorLibrary.h"
#include "NeuralBlock.h"
#include "NeuralLayer.h"




int main() {
    try {
        const int samples = 16, features = 4, batch = 1;
        Tensor X(samples, features, batch);
        Tensor Y(samples, 1, batch);

        for (int r = 0; r < samples; ++r) {
            int count = 0;
            for (int c = 0; c < features; ++c) {
                int bit = (r >> c) & 1;
                X(r, c, 0) = static_cast<float>(bit);
                count += bit;
            }
            Y(r, 0, 0) = (count % 2 == 1) ? 1.0f : 0.0f;
        }

        Neural l1(4, Activation_Type::GELU);
        Neural l2(8, Activation_Type::GELU);
        Neural l3(1, Activation_Type::SIGMOID);
        Tensor_Loss_Function loss(tensor_loss_function::CROSS_ENTROPY_LOSS, 1);

        NeuralBlock block(X, {l1, l2,l3}, {loss});

        const float lr = 0.05f;
        const int iterations = 5000;
        float avg_loss = block.Train(X, Y, lr, iterations, 250);

        std::cout << "\nAverage loss over " << iterations << " iterations: " << avg_loss << "\n\n";

        block.Get_Input_Mutable() = X;
        block.Forward_Pass();
        const Tensor& P = block.Get_Output_View();

        int correct = 0;
        std::cout << std::fixed << std::setprecision(4);
        for (int r = 0; r < samples; ++r) {
            int y = (Y(r, 0, 0) >= 0.5f) ? 1 : 0;
            float p = P(r, 0, 0);
            int pred = (p >= 0.5f) ? 1 : 0;
            if (pred == y) ++correct;

            std::cout << "x=";
            for (int c = features - 1; c >= 0; --c)
                std::cout << int(X(r, c, 0));
            std::cout << "  y=" << y << "  p=" << p << "  pred=" << pred << "\n";
        }

        std::cout << "\nAccuracy: " << correct << "/" << samples
                  << " (" << (100.0 * correct / samples) << "%)\n";


    }
    catch (const std::runtime_error &error) {
        std::cerr << "[Runtime Error] " << error.what() << "\n";
    }
    catch (...) {
        std::cerr << "[Unknown Error] Something went wrong.\n";
    }

    return 0;
}