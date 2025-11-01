# FLUX: A Deep Learning Framework Built from First Principles

[![Watch Demo](https://img.youtube.com/vi/2Sr-psavJNk/0.jpg)](https://www.youtube.com/watch?v=2Sr-psavJNk)

## What is FLUX?

FLUX is an educational deep learning framework that reimplements the core functionality of PyTorch and TensorFlow from scratch, using only C++ and the Standard Template Library. No external dependencies, no black boxes—just pure C++ implementing every matrix multiplication, every gradient calculation, and every optimization step manually.

This is not a production framework. It's an exercise in understanding neural networks by building them from the ground up. Every architectural decision, every memory layout, every backward pass is implemented explicitly to expose the mathematics and engineering that modern ML frameworks abstract away.

The framework is built around a custom **3D tensor library** that serves as the computational foundation for all model architectures—Multi-Layer Perceptrons, Convolutional Neural Networks, and Transformers. The tensor library handles all forward and backward operations with explicit gradient tracking, allowing any model configuration to be constructed and trained.

## Why Build This?

Modern deep learning frameworks make it trivially easy to build and train neural networks. You can define a model in a few lines of PyTorch and start training immediately. But this convenience comes at a cost: you never see how backpropagation actually works, how gradients flow through layers, or how memory is managed during training.

FLUX takes the opposite approach. By implementing everything manually, you are forced to understand:

**The Mathematics**: Writing backpropagation by hand reveals the chain rule in action. Computing `∂C/∂w = (∂C/∂a) × h(a) × (∂h/∂a)` isn't just an equation—it's code you write, debug, and optimize.

**The Engineering**: How should tensors be laid out in memory? When should you allocate? How do you avoid unnecessary copies? These questions don't arise when using PyTorch, but they're critical for understanding performance.

**The Architecture**: Why do Transformers use residual connections? Why does batch normalization help? When you implement these components yourself, their purpose becomes crystal clear.

This project exists to help people understand the internals of neural networks deeply, not to replace existing frameworks. It's a learning tool for anyone who wants to see beyond the abstractions.

## Project Status

FLUX is a work in progress, developed alongside my research Master's in Computer Science. This is an ambitious, long-term project that continuously evolves as I deepen my understanding of different architectures.

### Multi-Layer Perceptron (Fully Functional)

The MLP implementation is complete and battle-tested. It successfully trains on multiple datasets and has been validated with the 4-bit parity problem, demonstrating proper gradient flow and convergence. All components—forward propagation, backpropagation, gradient descent, activation functions, and loss computation—work correctly for arbitrary network depths.

### Transformers (Partially Implemented)

The core building blocks of the Transformer architecture are implemented: tokenization with Byte-Pair Encoding, token embeddings, fixed positional encodings, single-head and multi-head attention mechanisms, layer normalization, and residual connections. The feed-forward network component (which uses the MLP implementation) is ready but not yet integrated into a complete Transformer block. Full assembly and training loops are in progress.

### Convolutional Neural Networks (In Progress)

Most CNN components are implemented: convolutional layers with kernel operations, max and average pooling layers, padding strategies (VALID and SAME), and the im2col algorithm for efficient convolution. The primary missing piece is backpropagation through convolutional layers. Forward passes work correctly, but gradient computation for kernels and feature maps is under development.

## Technical Architecture

### The Tensor Library: Foundation of Everything

Unlike frameworks that build on optimized BLAS libraries, FLUX implements its own tensor operations from scratch. The **Tensor class** is the computational workhorse, representing 3D data structures with dimensions `[rows × columns × depth]`:

```cpp
class Tensor {
public:
    Tensor(int rows, int columns, int depth);
    
    // Element access
    float& operator()(int row, int col, int depth);
    
    // Core operations
    void Tensor_Add(const Tensor& other);
    void Tensor_Hadamard_Product(const Tensor& other);
    void Tensor_MatMul(Tensor& result, const Tensor& A, const Tensor& B);
    
    // Initialization
    void Tensor_Xavier_Uniform_MLP(int fan_in, int fan_out);
    void Tensor_Xavier_Uniform_Conv(int number_of_kernels);
    void Fill(float value);
    
    // Dimensions
    int rows() const;
    int columns() const;
    int depth() const;
};
```

The third dimension serves different purposes depending on the architecture:
- **MLPs**: Batch size (process multiple samples simultaneously)
- **CNNs**: Channels (RGB images or feature maps)
- **Transformers**: Batch size (multiple sequences in parallel)

This unified representation allows the same tensor operations to power all model types. A matrix multiplication in an MLP uses the same code as a matrix multiplication in a Transformer attention mechanism.

### Matrix Library: A Legacy Utility

The framework originally used a 2D matrix library, which is still present in the codebase for backward compatibility and simple utility operations. However, the tensor library has superseded it as the primary computational engine. The matrix library exists mainly for operations that don't require batch processing or when working with truly 2D data.

### Memory Management and RAII

FLUX follows strict C++ RAII (Resource Acquisition Is Initialization) principles. All tensors manage their own memory through `std::unique_ptr<float[]>`, ensuring automatic cleanup and preventing leaks. There are no external dependencies—no Python bindings, no third-party libraries, just the C++ Standard Library.

Memory is allocated once during network initialization:

```cpp
void Neural::Resize_Tensors(const Tensor& input_tensor) {
    const int S = input_tensor.rows();      // Sequence length / samples
    const int I = input_tensor.columns();   // Input features
    const int B = input_tensor.depth();     // Batch size
    const int N = number_of_neurons_;       // Output features
    
    // Allocate all tensors upfront
    input_           = Tensor(S, I, B);
    weight_          = Tensor(N, I, 1);     // Shared across batch
    bias_            = Tensor(1, N, 1);
    pre_activation_  = Tensor(S, N, B);
    post_activation_ = Tensor(S, N, B);
    output_          = Tensor(S, N, B);
    
    // Gradient tensors
    dL_dw_  = Tensor(N, I, 1);
    dL_db_  = Tensor(1, N, 1);
    dL_dz_  = Tensor(S, N, B);
    da_dz_  = Tensor(S, N, B);
    dL_dx_downstream_layer_error_ = Tensor(S, I, B);
}
```

During training, no dynamic allocation occurs. All computations reuse pre-allocated buffers, making the memory footprint predictable and stable.

## Supported Model Architectures

### Multi-Layer Perceptron

The MLP implementation demonstrates how to construct and train fully-connected networks. Here's the actual usage pattern from the main file:

```cpp
#include "NeuralBlock.h"
#include "tensor-library/TensorLibrary.h"

int main() {
    // Load your data into tensors
    Tensor input_data(100, 4, 1);    // 100 samples, 4 features, batch size 1
    Tensor target_data(100, 1, 1);   // 100 labels
    
    // Define the network architecture
    Neural_Block network;
    
    // Add layers sequentially
    network.Add_Layer(16, Activation_Type::LEAKY_RELU);  // Hidden layer: 16 neurons
    network.Add_Layer(8, Activation_Type::LEAKY_RELU);   // Hidden layer: 8 neurons
    network.Add_Layer(1, Activation_Type::SIGMOID);      // Output layer: 1 neuron
    
    // Connect input and target data
    network.Set_Input(input_data);
    network.Set_Target(target_data);
    
    // Initialize weights using Xavier initialization
    network.Initialize();
    
    // Train the network
    float learning_rate = 0.01f;
    int epochs = 10000;
    network.Train(learning_rate, epochs);
    
    // Make predictions
    Tensor test_input(1, 4, 1);
    Tensor prediction = network.Predict(test_input);
    
    return 0;
}
```

The `Neural_Block` class manages the entire network as a sequence of `Neural` layer objects. Each layer maintains its own tensors for weights, biases, activations, and gradients. Calling `Train()` automatically handles the forward pass through all layers, loss computation, backward pass, and weight updates.

**Layer Construction**: The input layer is implicit—its size is determined by the input tensor's feature dimension. If your input has 4 features, the first hidden layer automatically expects 4 inputs. This makes the API cleaner since you only specify hidden and output layer sizes.

**Activation Functions**: Each layer can use different activations: `LINEAR`, `RELU`, `LEAKY_RELU`, `SIGMOID`, or `TANH`. The derivatives are computed automatically during backpropagation.

**Training Process**: The `Train()` method runs the standard training loop: forward pass → loss calculation → backward pass → gradient descent update. Loss is computed using the specified loss function (MSE for regression, cross-entropy for classification).

### Convolutional Neural Networks

CNN layers are under active development. The forward pass infrastructure is complete:

```cpp
// Convolutional layer with 32 filters, 3×3 kernels
Convolutional_Layer conv1(input_tensor, 32, ConvActivationType::RELU, 
                          PaddingStrategy::SAME, stride=1);

// Max pooling with 2×2 window
Pooling_Layer pool1(conv1.output(), 2, stride=2, PoolingStrategy::MAX_POOL);
```

Convolutional layers support both VALID (no padding) and SAME (zero padding) strategies. The kernel class wraps tensors with convenience methods for weight initialization. Pooling layers implement both max and average pooling with configurable window sizes and strides.

The im2col algorithm is implemented for efficient convolution, transforming image patches into column vectors that can be multiplied by flattened kernels. This is the same technique used by Caffe and other frameworks.

**What's Missing**: Backpropagation through convolutional layers. Computing gradients with respect to kernels and propagating errors back through pooling layers requires careful index tracking, which is currently being implemented and tested.

### Transformers

Transformer components exist as modular building blocks:

```cpp
// Tokenization
Tokenizer tokenizer;
tokenizer.Train_BPE(corpus, num_merges=1000);
Tensor token_ids = tokenizer.Tokenize(text);

// Embeddings
TokenEmbedding embeddings(vocab_size, d_model=512);
Tensor embedded = embeddings.Forward(token_ids);

// Positional encoding
FixedPositionalEncoding pos_enc(max_seq_len=512, d_model=512);
Tensor encoded = pos_enc.Add_Positional_Encoding(embedded);

// Multi-head attention
MultiHeadAttention mha(d_model=512, num_heads=8);
Tensor attention_output = mha.Forward(encoded, encoded, encoded);

// Layer normalization
LayerNormalization ln(d_model=512);
Tensor normalized = ln.Forward(attention_output);
```

Each component is functional and tested independently. The Byte-Pair Encoding tokenizer implements the full BPE algorithm with vocabulary building and text encoding. Multi-head attention correctly splits the input into multiple heads, computes scaled dot-product attention for each head, and concatenates the results.

**What's Missing**: Full integration into a Transformer block. The feed-forward network (which uses the MLP implementation) needs to be assembled with attention layers, layer normalization, and residual connections into a complete encoder/decoder block. Training loops and sequence generation are not yet implemented.

## Mathematical Implementation

### Manual Backpropagation

Backpropagation is implemented manually, not with automatic differentiation. This is intentional. Understanding how gradients flow backward through a network is the entire point of this project.

For each layer, the backward pass computes three things:

**1. Gradient with respect to pre-activation**: This combines the upstream gradient from the next layer with the derivative of the activation function:

```cpp
// Element-wise: dL/dz = dL/da ⊙ h'(z)
dL_dz_ = dL_da_upstream_layer_error_;
dL_dz_.Tensor_Hadamard_Product(da_dz_);
```

**2. Gradient with respect to weights**: This is the outer product of the input and the pre-activation gradient:

```cpp
// Matrix multiply: dL/dW = x^T × dL/dz
Tensor_MatMul_Transpose_A(dL_dw_, input_, dL_dz_);
```

**3. Gradient with respect to bias**: Sum the pre-activation gradient across all samples:

```cpp
// Sum over samples: dL/db = Σ(dL/dz)
Tensor_Sum_Across_Samples(dL_db_, dL_dz_);
```

Finally, propagate the error to the previous layer:

```cpp
// Matrix multiply: dL/dx = dL/dz × W^T  
Tensor_MatMul_Transpose_B(dL_dx_downstream_layer_error_, dL_dz_, weight_);
```

This explicit approach makes the chain rule visible in the code. Each operation corresponds directly to a mathematical derivative. There's no magic, no computational graph, just careful application of calculus.

### Forward Propagation

The forward pass for each layer follows the standard neural network formula:

```cpp
void Neural::Forward_Pass(const Tensor& input) {
    // Store input for backward pass
    input_ = input;
    
    // Linear transformation: z = W^T × x + b
    Tensor_MatMul_Transpose_B(pre_activation_, input_, weight_);
    
    // Broadcast and add bias
    Tensor broadcasted_bias;
    Tensor_Broadcast_At_Depth(broadcasted_bias, bias_, pre_activation_.depth());
    pre_activation_.Tensor_Add(broadcasted_bias);
    
    // Apply activation function: a = h(z)
    Tensor_Activation_Function(post_activation_, pre_activation_, activation_type_);
    
    // Output becomes input to next layer
    output_ = post_activation_;
}
```

### Gradient Descent

Weight updates use vanilla stochastic gradient descent:

```cpp
void Neural::Update_Weights(float learning_rate) {
    // W_new = W_old - η × dL/dW
    Tensor gradient_step = dL_dw_;
    gradient_step.Tensor_Scalar_Multiply(learning_rate);
    weight_.Tensor_Subtract(gradient_step);
    
    // b_new = b_old - η × dL/db
    Tensor bias_step = dL_db_;
    bias_step.Tensor_Scalar_Multiply(learning_rate);
    bias_.Tensor_Subtract(bias_step);
}
```

More sophisticated optimizers (momentum, Adam, RMSprop) are planned but not yet implemented. The goal is to first ensure correctness with the simplest optimizer, then add complexity.

## Project Structure

```
FLUX/
├── model-architectures/
│   ├── multi-layer-perceptron/
│   │   ├── mlp-feed-forward/
│   │   │   ├── NeuralLayer.cpp         # Single layer implementation
│   │   │   ├── NeuralLayer.h
│   │   │   ├── NeuralBlock.cpp         # Full network container
│   │   │   └── NeuralBlock.h
│   │   └── main.cpp                     # Usage examples and tests
│   │
│   ├── transformers/
│   │   ├── tokenizer/                   # BPE tokenization
│   │   ├── token_embedding/             # Embedding lookup
│   │   ├── positional-embeddings/       # Position encoding
│   │   ├── single-attention-head/       # Attention mechanism
│   │   ├── multi-head-attention/        # Multi-head wrapper
│   │   ├── layer-normalization/         # Layer norm
│   │   ├── residual-stream/             # Skip connections
│   │   ├── batch-operations/            # Batch processing
│   │   └── main.cpp                     # Component tests
│   │
│   └── convolutional-neural-networks/
│       ├── convolutional-layer/         # Conv2D operations
│       ├── pooling-layer/               # Max/average pooling
│       ├── kernels/                     # Kernel weight management
│       ├── padding-functions/           # SAME/VALID padding
│       └── utility-migratory-functions/ # Im2col and helpers
│
├── tensor-library/
│   ├── TensorLibrary.h                  # Main computational engine
│   └── TensorLibrary.cpp
│
├── matrix-library/                       # Legacy 2D operations
│   ├── MatrixLibrary.h
│   └── MatrixLibrary.cpp
│
├── activation-functions/
│   ├── ActivationFunctions.h            # ReLU, Sigmoid, Tanh, etc.
│   └── ActivationFunctions.cpp
│
├── activation-function-derivatives/
│   ├── ActivationFunctionDerivatives.h  # Gradient computations
│   └── ActivationFunctionDerivatives.cpp
│
├── loss-functions/
│   ├── TensorLossFunctions.h            # MSE, Cross-entropy
│   └── TensorLossFunctions.cpp
│
└── web-visualizations/                   # Real-time training monitoring
    ├── data-loader/
    └── main.cpp                          # WebSocket server
```

The project is organized by model architecture, with shared computational libraries (tensors, activations, loss functions) at the root level. Each architecture has its own main file demonstrating usage patterns.

## Building and Running

### Prerequisites

- CMake 3.26 or higher
- C++23 compatible compiler (GCC 13+, Clang 16+, MSVC 2022+)
- No external libraries required (only C++ STL)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/FLUX.git
cd FLUX

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all targets
cmake --build .

# Run MLP examples
./MLP_main

# Run Transformer component tests
./Transformer_main

# Run visualization server (optional)
./WebVis_main
```

### CMake Targets

The build system creates several executables:

- `MLP_main`: Multi-Layer Perceptron examples and tests
- `Transformer_main`: Transformer component demonstrations
- `WebVis_main`: Real-time training visualization server

Additionally, the following static libraries are built:
- `NeuralCore`: Shared computational components (tensors, activations, loss functions)
- `MultiLayerPerceptron`: MLP-specific architecture
- `TransformerModel`: Transformer components
- `WebVisualization`: Monitoring and visualization tools

## Testing and Validation

The MLP implementation has been validated on several problems:

**4-bit Parity Problem**: A classic non-linearly separable problem that requires hidden layers. The network successfully learns the XOR-like patterns with 100% accuracy after training.

**Multi-Layer Convergence**: Networks with arbitrary depth (tested up to 10 layers) successfully converge on regression and classification tasks, demonstrating that gradients flow correctly through deep architectures.

**Gradient Checking**: Numerical gradient checking has been used during development to verify that analytical gradients match finite difference approximations, ensuring backpropagation correctness.

## Performance Characteristics

FLUX is not optimized for speed—it's optimized for clarity. That said, certain design decisions provide reasonable performance:

**Memory Layout**: Tensors use contiguous memory allocation with row-major ordering, providing good cache locality during matrix operations.

**Pre-allocation**: All tensors are allocated during initialization, not during training. This eliminates allocation overhead in the hot path.

**No Python Overhead**: Being pure C++ means no interpreter overhead, no GIL contention, and no Python-C++ boundary crossings.

**Future Optimization**: The code is structured to allow SIMD vectorization and GPU acceleration in the future, though these are not currently implemented.

## Visualization and Monitoring

A WebSocket-based visualization server allows real-time monitoring of training progress. It serializes tensor data (weights, gradients, activations) to a binary format and streams them to a web interface, where you can observe how the network evolves during training.

This is particularly useful for debugging—watching weight distributions and gradient magnitudes helps identify issues like vanishing gradients, exploding gradients, or dead neurons.

## What's Next

Development continues on several fronts:

**Completing CNNs**: Implementing backpropagation through convolutional and pooling layers is the immediate priority. Once this works, MNIST digit classification will serve as the validation test.

**Assembling Transformers**: Connecting all the existing components into a full Transformer block and implementing the training loop. The goal is to train a small language model on a simple dataset.

**Advanced Optimizers**: Implementing Adam, RMSprop, and momentum-based gradient descent to improve training dynamics.

**Regularization**: Adding dropout, batch normalization, and weight decay.

**Model Serialization**: Saving and loading trained models so networks don't need to be retrained from scratch.

This is a long-term, ongoing project. Progress happens incrementally as I balance research, coursework, and implementation. The focus remains on understanding, not speed of development.

## Technical Resources

Understanding the mathematics and implementation requires background knowledge:

**Backpropagation**: "Neural Networks and Deep Learning" by Michael Nielsen provides excellent intuition for how gradients flow through networks.

**Matrix Calculus**: "The Matrix Cookbook" is invaluable for understanding the derivatives used in gradient computation.

**Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017) is the foundational paper, but Jay Alammar's illustrated guide provides better intuition.

**CNN Mechanics**: CS231n lecture notes from Stanford explain convolution, pooling, and backpropagation through CNNs clearly.

## Philosophy and Goals

This project embodies a specific philosophy about learning: **understanding comes from building, not just using**. You can read papers about backpropagation, watch lectures, work through derivations—but until you've implemented it yourself, debugged gradient flow, and fixed numerical instability issues, you don't truly understand it.

FLUX is intentionally low-level and explicit. It doesn't use automatic differentiation, doesn't hide memory management, and doesn't abstract away computational details. This makes it harder to use than PyTorch, but that's the point. The difficulty is the learning opportunity.

The goal is not to create a production-ready framework. It's to create a tool for understanding—for anyone who wants to know how neural networks actually work beneath the high-level APIs.

---

**Project Status**: Active development alongside Master's research  
**Last Updated**: November 2025  
**License**: Open source for educational purposes