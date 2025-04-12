#include <emscripten/emscripten.h>
#include "Neural_Blocks/Neural_Blocks.h"
#include "Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include <vector>

// Global neural block for the parity network
static Neural_Block* g_parity_network = nullptr;
// Global trainer object
static Train_Block_by_Backpropagation* g_trainer = nullptr;

// Helper: convert integer to a LossFunction enum
static LossFunction intToLossFunction(int x) {
    switch(x) {
        case 0: return LossFunction::MSE_LOSS;
        case 1: return LossFunction::HUBER_LOSS;
        case 2: return LossFunction::CROSS_ENTROPY_LOSS;
        case 3: return LossFunction::MAE_LOSS;
        default: return LossFunction::MSE_LOSS; // fallback
    }
}

// Helper: convert integer to an ActivationType enum
static ActivationType intToActivationType(int x) {
    switch(x) {
        case 0: return ActivationType::RELU;
        case 1: return ActivationType::SIGMOID;
        case 2: return ActivationType::TANH;
        case 3: return ActivationType::LEAKY_RELU;
        case 4: return ActivationType::SWISH;
        case 5: return ActivationType::LINEAR;
        default: return ActivationType::RELU; // fallback
    }
}

// Generate 4-bit parity data
void generate4BitParityData(Matrix& inputData, Matrix& expectedOutput) {
    // Generate all possible 4-bit patterns and their parity
    for (int i = 0; i < 16; i++) {
        // Convert decimal i to 4-bit binary representation
        inputData(i, 0) = (i & 1) ? 1.0f : 0.0f;  // Least significant bit
        inputData(i, 1) = (i & 2) ? 1.0f : 0.0f;  // Second bit
        inputData(i, 2) = (i & 4) ? 1.0f : 0.0f;  // Third bit
        inputData(i, 3) = (i & 8) ? 1.0f : 0.0f;  // Most significant bit

        // Calculate parity (1 if odd number of 1s, 0 if even number of 1s)
        int count_ones = 0;
        for (int j = 0; j < 4; j++) {
            if (inputData(i, j) > 0.5f) count_ones++;
        }
        expectedOutput(i, 0) = (count_ones % 2) ? 1.0f : 0.0f;
    }
}

extern "C" {

// Creates a neural network with one hidden layer
EMSCRIPTEN_KEEPALIVE
void createCustomBlock(int inputRows, int inputCols,
                       int hiddenNeurons,
                       int outputRows, int outputCols,
                       int lossFuncInt) {
    // Clean up previous objects if they exist
    if (g_trainer != nullptr) {
        delete g_trainer;
        g_trainer = nullptr;
    }
    if (g_parity_network != nullptr) {
        delete g_parity_network;
    }

    // Create input and output matrices
    Matrix inputData(inputRows, inputCols, 0.0f);
    Matrix expectedOutput(outputRows, outputCols, 0.0f);

    // Generate actual 4-bit parity data
    generate4BitParityData(inputData, expectedOutput);

    // Convert loss function type
    LossFunction lossFunc = intToLossFunction(lossFuncInt);

    // Create the neural network with one hidden layer
    g_parity_network = new Neural_Block(
            inputData,
            {
                    Neural_Layer_Skeleton(hiddenNeurons, intToActivationType(3)),
                    Neural_Layer_Skeleton(outputCols, ActivationType::SIGMOID)
            },
            lossFunc,
            expectedOutput
    );
}

// Creates a neural network with multiple hidden layers
EMSCRIPTEN_KEEPALIVE
void createMultiLayerNetwork(int inputRows, int inputCols,
                             int numHiddenLayers,
                             int* hiddenLayerSizes,
                             int* hiddenLayerActivations,
                             int outputRows, int outputCols,
                             int lossFuncInt) {
    // Clean up previous objects if they exist
    if (g_trainer != nullptr) {
        delete g_trainer;
        g_trainer = nullptr;
    }
    if (g_parity_network != nullptr) {
        delete g_parity_network;
    }

    // Create input and output matrices
    Matrix inputData(inputRows, inputCols, 0.0f);
    Matrix expectedOutput(outputRows, outputCols, 0.0f);

    // Generate actual 4-bit parity data
    generate4BitParityData(inputData, expectedOutput);

    // Convert loss function type
    LossFunction lossFunc = intToLossFunction(lossFuncInt);

    // Create the layer list
    std::vector<Neural_Layer_Skeleton> layers;
    for (int i = 0; i < numHiddenLayers; i++) {
        layers.push_back(Neural_Layer_Skeleton(
                hiddenLayerSizes[i],
                intToActivationType(hiddenLayerActivations[i])
        ));
    }

    // Add output layer with sigmoid activation
    layers.push_back(Neural_Layer_Skeleton(outputCols, ActivationType::SIGMOID));

    // Create neural network using the vector constructor
    g_parity_network = new Neural_Block(inputData, layers, lossFunc, expectedOutput);
}

// Get the number of layers in the network
EMSCRIPTEN_KEEPALIVE
int getBlockSize() {
    if (g_parity_network == nullptr) {
        return 0;
    }
    return g_parity_network->Get_Block_Size();
}

// Get loss value after forward pass
EMSCRIPTEN_KEEPALIVE
float getNetworkLoss() {
    if (g_parity_network == nullptr) {
        return -1.0f;
    }
    g_parity_network->Forward_Pass_With_Activation();
    return g_parity_network->Get_Block_Loss();
}

// Initialize the network trainer with specified learning rate
EMSCRIPTEN_KEEPALIVE
void Initialize_Trainer_For_One_Step_Iteration(float learningRate) {
    if (g_parity_network == nullptr) {
        return;
    }
    // Clean up previous trainer if it exists
    if (g_trainer != nullptr) {
        delete g_trainer;
    }
    // Create a new trainer with 0 iterations (since we'll train step by step)
    g_trainer = new Train_Block_by_Backpropagation(*g_parity_network, 0, learningRate);
}

// Execute a single training iteration
EMSCRIPTEN_KEEPALIVE
void Train_Network_By_One_Iteration() {
    // Train the network for one iteration
    if (g_trainer != nullptr) {
        g_trainer->Train_by_Backpropagation_One_Iteration();
    }
}

// Get current loss value
EMSCRIPTEN_KEEPALIVE
float Get_Block_Loss() {
    if (g_parity_network != nullptr) {
        return g_parity_network->Get_Block_Loss();
    }
    return -1.0f;
}

// Get predictions after training
EMSCRIPTEN_KEEPALIVE
void getPredictions(float* outputArray) {
    if (g_trainer == nullptr || g_parity_network == nullptr) {
        return;
    }

    // Get predictions from the trainer
    Matrix predictions = g_trainer->Get_Predictions();

    // Copy predictions to the output array
    for (int i = 0; i < 16; i++) {
        outputArray[i] = predictions(i, 0);
    }
}

// Get input patterns for verification
EMSCRIPTEN_KEEPALIVE
void getInputPatterns(float* outputArray) {
    if (g_parity_network == nullptr) {
        return;
    }

    // Get input matrix from the neural block
    Matrix inputMatrix = g_parity_network->Get_Block_Input_Matrix();

    // Copy input patterns to output array (4 bits x 16 patterns = 64 values)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            outputArray[i * 4 + j] = inputMatrix(i, j);
        }
    }
}

// Get target values for verification
EMSCRIPTEN_KEEPALIVE
void getTargetValues(float* outputArray) {
    if (g_parity_network == nullptr) {
        return;
    }

    // Get target matrix from the neural block
    const Matrix& targetMatrix = g_parity_network->Get_Block_Target_Matrix();

    // Copy target values to output array
    for (int i = 0; i < 16; i++) {
        outputArray[i] = targetMatrix(i, 0);
    }
}

} // extern "C"