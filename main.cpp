#include "Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include "Loss_Functions/Loss_Functions.h"
#include <iomanip>

int main() {
    // Create input data for 4-bit parity problem
    // All 16 possible combinations of 4 bits
    Matrix input_data(16, 4);
    Matrix expected_output(16, 1);

    // Generate all possible 4-bit patterns and their parity
    for (int i = 0; i < 16; i++) {
        // Convert decimal i to 4-bit binary representation
        input_data(i, 0) = (i & 1) ? 1.0f : 0.0f;
        input_data(i, 1) = (i & 2) ? 1.0f : 0.0f;
        input_data(i, 2) = (i & 4) ? 1.0f : 0.0f;
        input_data(i, 3) = (i & 8) ? 1.0f : 0.0f;

        // Calculate parity (1 if odd number of 1s, 0 if even number of 1s)
        int count_ones = 0;
        for (int j = 0; j < 4; j++) {
            if (input_data(i, j) > 0.5f) count_ones++;
        }
        expected_output(i, 0) = (count_ones % 2) ? 1.0f : 0.0f;
    }

    // Create neural network for 4-bit parity
    std::cout << "Creating neural network for 4-bit parity problem...\n";

    // Define network architecture with two hidden layers
    Neural_Block parity_network(
            input_data,
            {
                    Neural_Layer_Skeleton(4, ActivationType::SIGMOID),  // First hidden layer// Second hidden layer
                    Neural_Layer_Skeleton(1, ActivationType::SIGMOID)   // Output layer
            },
            LossFunction::CROSS_ENTROPY_LOSS,
            expected_output
    );

    // Print initial state
    parity_network.Forward_Pass_With_Activation();
    parity_network.Calculate_Block_Loss();
    std::cout << "Initial loss: " << parity_network.Get_Loss() << "\n\n";

    // Train the network
    std::cout << "Training the network...\n";
    int iterations = 100;  // More iterations needed for this complex problem
    float learning_rate = 0.1f;

    // Create training object and train
    Train_Block_by_Backpropagation trainer(parity_network, iterations, learning_rate);

    // Perform forward pass to get final results
    parity_network.Forward_Pass_With_Activation();

    // Get predictions
    Matrix predictions = trainer.Get_Predictions();

    // Calculate final loss
    parity_network.Calculate_Block_Loss();
    std::cout << "Final loss: " << parity_network.Get_Loss() << "\n\n";

    // Print results
    std::cout << "Testing the trained 4-bit parity network:\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Inputs | Predicted | Expected\n";
    std::cout << "---------------------------------------\n";

    int correct_predictions = 0;
    for (int i = 0; i < 16; i++) {
        std::cout << (input_data(i, 0)) << " "
                  << (input_data(i, 1)) << " "
                  << (input_data(i, 2)) << " "
                  << (input_data(i, 3)) << " | "
                  << std::fixed << std::setprecision(4) << predictions(i, 0) << " | "
                  << expected_output(i, 0) << "\n";

        // Count correct predictions (rounded)
        if (std::round(predictions(i, 0)) == expected_output(i, 0)) {
            correct_predictions++;
        }
    }

    // Calculate accuracy
    float accuracy = static_cast<float>(correct_predictions) / 16.0f * 100.0f;
    std::cout << "\nAccuracy: " << accuracy << "%\n";

    std::cout << "\n4-bit parity implementation completed successfully!\n";
    return 0;
}