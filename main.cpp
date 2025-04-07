/**
 * @file main.cpp
 * @brief Neural network demonstration solving the 4-bit parity problem
 * @author Rakib
 * @date 2025-04-07
 *
 * This program demonstrates a neural network learning to solve the 4-bit parity problem.
 * The parity problem requires determining whether a binary pattern has an odd or even
 * number of 1s. It's a classic problem for neural networks because it's not linearly
 * separable, requiring a multi-layer network with nonlinear activation functions.
 */

#include "Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include "Loss_Functions/Loss_Functions.h"
#include "Utility_Functions/Utility_Functions.h"
#include <iomanip>
#include <iostream>

int main() {
    std::cout << "===================================================\n";
    std::cout << "           4-BIT PARITY NEURAL NETWORK             \n";
    std::cout << "===================================================\n\n";

    //==========================================================================
    // STEP 1: Prepare training data
    //==========================================================================
    std::cout << "Preparing training data...\n";

    // Create matrices for input data and expected outputs
    Matrix input_data(16, 4);         // 16 samples (all possible 4-bit patterns), 4 features (bits)
    Matrix expected_output(16, 1);    // 16 corresponding parity values (0 or 1)

    // Generate all possible 4-bit patterns and their parity
    for (int i = 0; i < 16; i++) {
        // Convert decimal i to 4-bit binary representation
        input_data(i, 0) = (i & 1) ? 1.0f : 0.0f;  // Least significant bit
        input_data(i, 1) = (i & 2) ? 1.0f : 0.0f;  // Second bit
        input_data(i, 2) = (i & 4) ? 1.0f : 0.0f;  // Third bit
        input_data(i, 3) = (i & 8) ? 1.0f : 0.0f;  // Most significant bit

        // Calculate parity (1 if odd number of 1s, 0 if even number of 1s)
        int count_ones = 0;
        for (int j = 0; j < 4; j++) {
            if (input_data(i, j) > 0.5f) count_ones++;
        }
        expected_output(i, 0) = (count_ones % 2) ? 1.0f : 0.0f;
    }

    // Display a few samples to verify data generation
    std::cout << "Sample training data (first 5 samples):\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "| Binary Pattern | Parity (odd=1, even=0) |\n";
    std::cout << "---------------------------------------------------\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "| " << input_data(i, 0) << " " << input_data(i, 1) << " "
                  << input_data(i, 2) << " " << input_data(i, 3) << " | "
                  << expected_output(i, 0) << " |\n";
    }
    std::cout << "---------------------------------------------------\n\n";

    //==========================================================================
    // STEP 2: Create the neural network
    //==========================================================================
    std::cout << "Creating neural network for 4-bit parity problem...\n";

    // Network architecture:
    // - Input layer: 4 neurons (one per bit)
    // - First hidden layer: 4 neurons with LeakyReLU activation
    // - Second hidden layer: 8 neurons with LeakyReLU activation
    // - Output layer: 1 neuron with Sigmoid activation (for binary classification)
    Neural_Block parity_network(
            input_data,
            {
                    Neural_Layer_Skeleton(4, ActivationType::LEAKY_RELU),
                    Neural_Layer_Skeleton(8, ActivationType::LEAKY_RELU),
                    Neural_Layer_Skeleton(1, ActivationType::SIGMOID)
            },
            LossFunction::CROSS_ENTROPY_LOSS,
            expected_output
    );

    std::cout << "Network architecture:\n";
    std::cout << "- Input layer: 4 neurons\n";
    std::cout << "- Hidden layer 1: 4 neurons (LeakyReLU)\n";
    std::cout << "- Hidden layer 2: 8 neurons (LeakyReLU)\n";
    std::cout << "- Output layer: 1 neuron (Sigmoid)\n";
    std::cout << "- Loss function: Binary Cross-Entropy\n\n";

    //==========================================================================
    // STEP 3: Train the network
    //==========================================================================
    std::cout << "Training the network...\n";

    // Training hyperparameters
    int iterations = 50000;       // Number of training iterations
    float learning_rate = 0.01f;  // Learning rate for gradient descent

    std::cout << "Training parameters:\n";
    std::cout << "- Iterations: " << iterations << "\n";
    std::cout << "- Learning rate: " << learning_rate << "\n\n";

    // Create trainer and start the training process
    // Note: Training begins automatically when the Train_Block_by_Backpropagation object is created
    Train_Block_by_Backpropagation trainer(parity_network, iterations, learning_rate);

    // Ensure network is fully forward propagated to get final results
    parity_network.Forward_Pass_With_Activation();

    // Get predictions for evaluation
    Matrix predictions = trainer.Get_Predictions();

    std::cout << "Training completed!\n";
    std::cout << "Final loss: " << parity_network.Get_Block_Loss() << "\n\n";

    //==========================================================================
    // STEP 4: Evaluate the trained network
    //==========================================================================
    std::cout << "Evaluating the trained network...\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "| Binary Pattern | Predicted Probability | Expected Output | Correct? |\n";
    std::cout << "------------------------------------------------------------------------\n";

    int correct_predictions = 0;
    for (int i = 0; i < 16; i++) {
        // Round prediction to get binary output (0 or 1)
        float predicted_probability = predictions(i, 0);
        int rounded_prediction = std::round(predicted_probability);
        bool is_correct = (rounded_prediction == expected_output(i, 0));

        // Update correct prediction count
        if (is_correct) {
            correct_predictions++;
        }

        // Display results with indicating correct/incorrect predictions
        std::cout << "| " << input_data(i, 0) << " " << input_data(i, 1) << " "
                  << input_data(i, 2) << " " << input_data(i, 3) << " | "
                  << std::fixed << std::setprecision(4) << std::setw(8) << predicted_probability << "        | "
                  << std::setw(7) << expected_output(i, 0) << "        | "
                  << (is_correct ? "  YES  " : "   NO  ") << " |\n";
    }
    std::cout << "------------------------------------------------------------------------\n";

    // Calculate and display accuracy
    float accuracy = static_cast<float>(correct_predictions) / 16.0f * 100.0f;
    std::cout << "\nFinal accuracy: " << accuracy << "% ("
              << correct_predictions << " correct out of 16 patterns)\n\n";

    std::cout << "===================================================\n";
    std::cout << "       4-BIT PARITY IMPLEMENTATION COMPLETE        \n";
    std::cout << "===================================================\n";

    return 0;
}