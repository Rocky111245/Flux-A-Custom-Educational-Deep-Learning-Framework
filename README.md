# FLUX: A Custom-Made Deep Learning Framework

## Introduction

In the world of deep learning, most frameworks abstract away the complexities, making it easy to build and deploy models. However, this convenience often comes at the cost of understanding. This framework flips the script by exposing the fundamental principles of neural networks. By building everything from scratch, including the matrix operations and gradient calculations, you gain an unparalleled insight into how neural networks function at a mathematical level.

This custom Deep Neural Network Framework, developed exclusively in C and C++, serves as both an educational tool and a robust platform for deep learning. It is designed to provide a clear view into the inner workings of neural networks, allowing users to comprehend and manipulate every aspect of the learning process.

Key features of this framework include:

1. **Custom Matrix Library**: A hand-crafted library for efficient matrix operations, optimized specifically for neural network computations.

2. **Transparent Gradient Calculation**: Unlike traditional frameworks that use Directed Acyclic Graphs, this framework implements a unique gradient calculation engine, providing clarity in the differentiation process.

3. **Flexible Network Architecture**: Users have the ability to design and implement various network architectures, from simple perceptrons to complex, deep networks.

4. **Comprehensive Visualization Tools**: Built-in tools for visualizing network structure and training progress, enhancing the educational value of the framework.

5. **Pure C/C++ Implementation**: Leveraging the efficiency of low-level programming to provide insights into performance optimization in machine learning.

This framework is ideal for students seeking to grasp the fundamentals of neural networks, researchers exploring novel architectures, and developers aiming to deepen their understanding of deep learning principles. By providing a transparent view into every calculation and decision made by the network, it bridges the gap between theoretical knowledge and practical implementation in the field of artificial intelligence.

Watch the video on my youtube channel.

[![Watch the video](https://img.youtube.com/vi/2Sr-psavJNk/0.jpg)](https://www.youtube.com/watch?v=2Sr-psavJNk)


## Framework Architecture

The FLUX neural network framework is designed around a custom matrix mathematical library. The core idea is to manage and process data in the form of matrices, allowing for efficient and scalable operations across the layers of the neural network.

### Core Components

Each neural layer in the FLUX framework consists of several matrices that hold critical information:
- **weights_matrix**: Holds the weights_matrix of every connection within the layer.
- **bias_matrix**: Stores the biases associated with each connection.
- **input_matrix**: Captures the input values received by each neuron in the layer.
- **weight_and_input_matrix**: Represents the result after the input values are multiplied by their corresponding weights_matrix.
- **weight_input_bias_matrix**: This matrix holds the values after adding the biases to the weighted inputs.
- **activated_output_matrix**: Stores the activated output of the neurons in the layer after applying the activation function.
- **dC_dy_matrix**: Holds the derivative of the cost function with respect to the experimental output.
- **dC_da_matrix**: Contains the derivative of the cost function with respect to the input values.
- **dC_dw_matrix**: Stores the derivative of the cost function with respect to the weights_matrix.
- **dC_db_matrix**: Captures the derivative of the cost function with respect to the biases.
- **dh_da_matrix**: Holds the derivative of the activation function with respect to its input.

### Layer Structure

In FLUX, the input layer is implicit, meaning the number of neurons in the first layer is determined by the number of parameters in the input data. For instance, if the input has 4 parameters, the first layer will have 4 neurons.

Given a specified structure like `8-4-1`, the actual structure in FLUX would be `4(implicit)-8-4-1`. Here’s how it works:
- **First layer (implicit)**: The number of neurons is automatically set to match the number of input parameters.
- **Subsequent layers**: These are explicitly defined by the user (e.g., 8-4-1).

### Class Configuration

Each neural layer is represented by the `Neural_Layer` class, which encapsulates all the matrices described above. When constructing a neural network, each connection between layers is modeled as an instance of this class. For a network with `n` layers, there will be `n-1` instances of `Neural_Layer` corresponding to the connections between each pair of layers.

For example, in a network structured as `4-8-4-1`, the framework will manage three instances of `Neural_Layer` to handle the connections between the input layer and the first hidden layer, between the first hidden layer and the second hidden layer, and between the second hidden layer and the output layer.

This architecture allows for a clear and modular approach to neural network design, making it easier to modify, extend, and optimize the framework according to the user's needs.

## Key Functions and Their Descriptions

### Utility Functions

1. **Matrix_Data_Preprocessor**  
   **Description**: Preprocesses raw input data into a matrix format that the neural network can use. It takes parameters like the number of rows, columns, start, and end positions of the data and returns a formatted matrix.
   
   **Example Usage**:
   ```cpp
   Matrix input_matrix = Matrix_Data_Preprocessor(100, 4, 0, 0, your_input_data);
   ```
   
2. **Matrix_Multiply**  
   **Description**: Multiplies two matrices and stores the result in a third matrix. Essential for the forward pass where the input data is multiplied by the weight matrix.
   
   **Example Usage**:
   ```cpp
   Matrix_Multiply(weight_and_input_matrix, input_matrix, weights_matrix);
   ```

3. **Matrix_Broadcast**  
   **Description**: Broadcasts a smaller matrix (like a bias) across a larger matrix by duplicating its values. Useful when adding biases to the weighted sum in neural networks.
   
   **Example Usage**:
   ```cpp
   Matrix_Broadcast(Broadcasted_Bias, bias_matrix, rows, columns);
   ```

4. **Matrix_Add**  
   **Description**: Adds two matrices element-wise and stores the result in another matrix. Used frequently after matrix multiplication to add biases.
   
   **Example Usage**:
   ```cpp
   Matrix_Add(weight_input_bias_matrix, weight_and_input_matrix, Broadcasted_Bias);
   ```

5. **Matrix_Transpose**  
   **Description**: Computes the transpose of a matrix. Important during backpropagation when the gradients are being calculated.
   
   **Example Usage**:
   ```cpp
   Matrix_Transpose(transposed_matrix, original_matrix);
   ```

6. **Matrix_Hadamard_Product**  
   **Description**: Performs element-wise multiplication (Hadamard product) between two matrices. Essential during backpropagation for gradient calculation.
   
   **Example Usage**:
   ```cpp
   Matrix_Hadamard_Product(result_matrix, matrix_a, matrix_b);
   ```

7. **Matrix_Scalar_Multiply**  
   **Description**: Multiplies each element of a matrix by a scalar value. Useful for scaling gradients during the update step.
   
   **Example Usage**:
   ```cpp
   Matrix_Scalar_Multiply(matrix, scalar);
   ```


### Main Functions

1. **Form_Network**  
   **Description**: This function is the backbone of the FLUX framework. It initializes the entire neural network by creating each layer with specified neurons, activation functions, and connections between layers.

   **Detailed Explanation**:
   - **Parameters**:
     - `layers`: An initializer list specifying the number of neurons in each layer.
     - `inputMatrix`: The input data matrix.
     - `outputMatrix`: The expected output data matrix.
     - `hidden_activation`: The activation function for hidden layers (e.g., Leaky ReLU).
     - `output_activation`: The activation function for the output layer (e.g., Sigmoid).

   - **Process**:
     - It forms the 'skeleton' of the network. It initializes empty matrices which will later be used in the forward pass and back propagation.
     - The function first initializes the network structure by determining the size of each layer based on the provided initializer list.
     - It then iterates through each layer, initializing the weights_matrix, biases, and activation functions.
     - The input matrix is passed to the first layer, and each subsequent layer takes the output of the previous layer as its input.
     - Finally, it returns a `Neural_Layer_Information` object, which contains all the necessary details about the layers, their connections, and the matrices used for computation.

   **Example Usage**:
   ```cpp
   auto neural_network = Form_Network({8, 4, 1}, input_matrix, target_matrix, 
                                      ActivationType::LEAKY_RELU, ActivationType::SIGMOID);
   ```


2. **Forward_Pass**  
   **Description**: Executes the forward propagation through the network. This involves computing the weighted sums for each layer, applying activation functions, and passing the results to the next layer.

   **Detailed Explanation**:
   - **Process**:
     - The function starts with the input matrix and multiplies it by the weights_matrix of the first layer.
     - The biases are then added to the result of the multiplication.
     - The result is passed through the specified activation function (e.g., ReLU, Sigmoid).
     - This process is repeated for each layer in the network until the final output is produced.

   **Example Usage**:
   ```cpp
   Forward_Pass(neural_network);
   ```



3. **Back_Propagation**  
   **Description**: Implements the backpropagation algorithm to calculate gradients and update the weights_matrix and biases of the network. This is the core of the learning process in neural networks.

   **Detailed Explanation**:
   - **Parameters**:
     - `neural_layer_information`: A `Neural_Layer_Information` object that holds all the necessary details about the network layers, including the matrices for weights_matrix, biases, inputs, and outputs.
     - `mean_squared_error`: A reference to a float variable where the computed mean squared error will be stored.
   
   - **Process**:
     - The function first computes the loss (e.g., Mean Squared Error) by comparing the network's output with the expected output.
     - It then calculates the gradient of the loss with respect to the output using the derivative of the activation function.
     - These gradients are propagated backward through the network, layer by layer, using the chain rule of calculus.
     - The gradients with respect to the weights_matrix and biases are stored and later used to update these parameters.
   
   - **Importance of the Matrix Library**:
     - The backpropagation process heavily relies on the custom matrix library, which handles all the matrix operations efficiently.
     - Calculations such as matrix multiplication, element-wise operations (Hadamard product), and transpositions are performed using this library, ensuring that the gradients are computed accurately and efficiently.
     - The modular design of the matrix library makes it possible to perform complex calculations seamlessly, which is crucial for the backpropagation algorithm's performance.

   **Example Usage**:
   ```cpp
   Back_Propagation(neural_network, mean_squared_error);
   ```



4. **Learn**  
   **Description**: The main training loop of the neural network. It repeatedly calls the forward pass, backpropagation, and parameter update functions over a specified number of iterations.

   **Detailed Explanation**:
   - **Parameters**:
     - `neural_layer_information`: The information about the network layers, weights_matrix, biases, etc.
     - `learning_rate`: The rate at which the network updates its parameters.
     - `iterations`: The number of times the network should iterate over the training data.

   - **Process**:
     - The function first initializes the network and calculates the initial Mean Squared Error (MSE).
     - It then enters a loop where it performs forward propagation, backpropagation, and updates the weights_matrix and biases based on the gradients.
     - The MSE is tracked over the iterations to monitor the training progress.
     - The final trained network and the MSE graph are outputted.

   **Example Usage**:
   ```cpp
   Learn(neural_network, 0.01, 450000);
   ```


5. **Activate**  
   **Description**: Applies the specified activation function to the weighted sum of inputs and biases in each neuron of a layer. This is crucial for introducing non-linearity into the network.

   **Detailed Explanation**:
   - **Parameters**:
     - `activation_type`: The type of activation function to apply (e.g., ReLU, Sigmoid).
   - **Process**:
     - The function iterates through the matrix of weighted sums and applies the chosen activation function to each element.

   **Example Usage**:
   ```cpp
   neural_layers[i].Activate(activation_type);
   ```


6. **Dh_Da_Function**  
   **Description**: Computes the derivative of the activation function with respect to its input, which is essential for backpropagation.

   **Detailed Explanation**:
   - **Parameters**:
     - `is_last_layer`: A boolean flag indicating if the current layer is the output layer.
   - **Process**:
     - The function calculates the derivative of the activation function for each neuron in the layer.
     - These derivatives are used during backpropagation to calculate the necessary gradients for updating the network.

   **Example Usage**:
   ```cpp
   neural_layers[i].Dh_Da_Function(is_last_layer);
   ```

### Putting It All Together

These functions work in harmony to create, train, and evaluate a deep learning model using the FLUX framework. By understanding how each function contributes to the overall process, users can modify and extend the framework to suit their specific needs.

---

# Comprehensive Mathematical Derivation of Neural Network Gradient Computation

## 1. Introduction to Gradient-Based Learning in Neural Networks

In the realm of neural networks, the process of learning is fundamentally an optimization problem. The network aims to minimize a cost function, which quantifies the discrepancy between the network's predictions and the actual target values. The primary tool for this optimization is gradient descent, which relies on computing the gradients of the cost function with respect to the network's parameters (weights_matrix and biases).

The backpropagation algorithm, which efficiently computes these gradients, is built upon the chain rule of calculus. This section will delve into the mathematical foundations of this process, deriving the key equations that drive learning in neural networks.

## 2. Fundamental Equation of Gradient Computation

### 2.1 The Chain Rule in Neural Networks

The cornerstone of gradient computation in neural networks is the chain rule of calculus. For a neural network, this rule allows us to propagate the error gradient backwards through the layers. The fundamental equation for computing gradients is:

```math
\frac{\partial C}{\partial w_i} = \frac{\partial C}{\partial Y_i} \times \frac{\partial Y_i}{\partial w_i}
```

Where:
- C is the cost function
- Y_i is the output of a neuron
- w_i is a weight connecting to that neuron

This equation encapsulates a crucial concept: the sensitivity of the cost to a weight (∂C/∂w_i) is a product of two factors:
1. How sensitive the cost is to the neuron's output (∂C/∂Y_i)
2. How sensitive the neuron's output is to the weight (∂Y_i/∂w_i)

### 2.2 Significance in Neural Network Learning

This decomposition is pivotal because it allows the learning algorithm to:
1. Reuse computations: ∂C/∂Y_i can be reused for all weights_matrix feeding into the same neuron.
2. Implement the backpropagation algorithm efficiently: gradients are computed layer by layer, from output to input.

## 3. Linear Combination in Neural Networks

### 3.1 A Simple Linear Neuron Model

Consider a simple linear combination within a neuron:

```math
Y_i = 3w_1 + 4w_2
```

Here, 3 and 4 are input values (often denoted as a1 and a2) obtained from the previous layer during forward propagation. This linear combination is a fundamental operation in neural networks, forming the basis of the weighted sum before activation.

### 3.2 Derivation of Partial Derivatives

For this linear combination, we can derive the partial derivatives:

```math
\frac{\partial Y_i}{\partial w_1} = 3
```

```math
\frac{\partial Y_i}{\partial w_2} = 4
```

These derivatives represent the sensitivity of the neuron's output to each weight. Notably, they are constant and equal to the input values, which is a characteristic of linear functions.

### 3.3 Gradient Computation

Applying our fundamental equation, we get:

```math
\frac{\partial C}{\partial w_1} = \frac{\partial C}{\partial Y_i} \times 3
```

```math
\frac{\partial C}{\partial w_2} = \frac{\partial C}{\partial Y_i} \times 4
```

### 3.4 Interpretation and Implications

This result reveals several key insights:

1. The gradient of the cost with respect to a weight is proportional to the corresponding input.
2. Larger inputs lead to larger gradients, causing those weights_matrix to be updated more aggressively during learning.
3. The term ∂C/∂Y_i, common to both equations, represents how the cost changes with the neuron's output. This term is propagated from later layers in the backpropagation process.

## 4. Activation Functions and Non-linearity

### 4.1 Necessity of Activation Functions

While linear combinations are computationally simple, neural networks derive their power from non-linear activation functions. These functions introduce non-linearity, allowing networks to approximate complex, non-linear relationships in data.

### 4.2 Gradient Computation with Activation Functions

When an activation function is introduced, our gradient computation becomes more intricate. Consider a connection from Node A to Node B, with an activation function h applied at Node A:

```math
\frac{\partial C}{\partial w_A} = \left(\frac{\partial C}{\partial a_B}\right) \times (h_A) \times \left(\frac{\partial h_A}{\partial a_A}\right)
```

Where:
- ∂C/∂a_B is the gradient at Node B (the next layer)
- h_A is the output of the activation function at Node A
- ∂h_A/∂a_A is the derivative of the activation function

### 4.3 Component Analysis

This equation can be broken down into three crucial components:

1. Global Derivative (∂C/∂a_B): This term propagates the error gradient from the subsequent layer, embodying the chain rule across layers.

2. Activation Output (h_A): This represents the actual output of the neuron after applying the activation function. Its inclusion ensures that the weight update is proportional to the neuron's activity.

3. Local Derivative (∂h_A/∂a_A): This term captures the local sensitivity of the activation function. It's critical for overcoming the limitations of linear models and allows the network to learn non-linear relationships.

### 4.4 Implications for Network Behavior

This formulation has profound implications for network behavior:

1. Vanishing/Exploding Gradients: Depending on the activation function, ∂h_A/∂a_A can become very small (vanishing gradient) or very large (exploding gradient), affecting the network's ability to learn.

2. Feature Detection: The inclusion of h_A in the gradient means that weights_matrix connected to highly activated neurons are updated more, reinforcing the detection of important features.

3. Non-linear Capacity: The non-linear activation function allows each neuron to approximate a small non-linear function, with the network as a whole approximating highly complex non-linear mappings.

### 4.5 Detailed Breakdown of Neural Network Gradient Computation

To fully understand the role of activation functions in neural networks, it's crucial to examine the gradient computation equation in detail. This equation is fundamental to the backpropagation algorithm, which is how neural networks learn.

#### The Equation: ∂C/∂w_A = (∂C/∂a_B) × (h_A) × (∂h_A/∂a_A)

Let's break down each component:

1. **∂C/∂w_A (Left side of the equation)**
   - **Meaning**: The rate of change of the cost function C with respect to a weight w_A in the network.
   - **Significance**: Indicates how much a small change in w_A would affect the overall network error. This is the value we compute to update weights_matrix during training.

2. **(∂C/∂a_B) (First term on right side)**
   - **Meaning**: The rate of change of the cost function C with respect to the activation a_B of the next layer (Node B).
   - **Significance**: Propagates the error gradient from subsequent layers, embodying the chain rule across layers.
   - **In practice**: Computed during the backward pass, starting from the output layer and moving towards the input layer.

3. **(h_A) (Second term on right side)**
   - **Meaning**: The output of the activation function at Node A.
   - **Significance**: Ensures weight updates are proportional to neuron activity. Highly active neurons (large h_A) have their weights_matrix updated more aggressively.
   - **In practice**: Computed during the forward pass and stored for use in backpropagation.

4. **(∂h_A/∂a_A) (Third term on right side)**
   - **Meaning**: The derivative of the activation function at Node A with respect to its input.
   - **Significance**: Captures the local sensitivity of the activation function, crucial for introducing non-linearity.
   - **In practice**: The form depends on the chosen activation function (e.g., sigmoid, ReLU, tanh).

#### The Big Picture

This equation encapsulates key principles of neural network learning:

1. **Chain Rule Application**: Efficiently computes complex derivatives.
2. **Error Propagation**: Allows errors from later layers to influence earlier layer weight updates.
3. **Activity-Based Learning**: More active neurons have a larger impact on learning.
4. **Non-linear Capacity**: Introduces non-linearity, enabling complex relationship learning.

#### Additional Implications for Network Behavior

1. **Vanishing/Exploding Gradients**: Extreme values of ∂h_A/∂a_A can lead to learning difficulties.
2. **Feature Detection**: Highly activated neurons' weights_matrix are updated more, reinforcing important feature detection.
3. **Learning Dynamics**: The interplay between terms creates complex learning dynamics.
4. **Activation Function Choice**: Affects ∂h_A/∂a_A, impacting the overall learning process.

Understanding this equation and its components is crucial for effective neural network design, appropriate activation function selection, and troubleshooting training issues.

### 4.6 Practical Considerations in Choosing Activation Functions

When selecting activation functions for a neural network, consider the following:

1. **Problem Type**: For binary classification, sigmoid might be suitable for the output layer, while ReLU or its variants often work well for hidden layers in various tasks.

2. **Network depth**: Deeper networks may benefit from activation functions that mitigate vanishing gradients, such as ReLU or its variants.

3. **Computational Efficiency**: ReLU is computationally efficient, making it a popular choice for large networks.

4. **Desired Properties**: Consider whether you need bounded outputs (sigmoid, tanh) or unbounded outputs (ReLU, Leaky ReLU).

5. **Potential Issues**: Be aware of issues like the dying ReLU problem and consider alternatives like Leaky ReLU if necessary.

6. **Empirical Performance**: Often, the best approach is to experiment with different activation functions and evaluate their performance on your specific task.

By carefully considering these factors and understanding the gradient computation process, you can make informed decisions about activation functions in your neural network designs.

## 5. Multiplicative Interactions in Neural Networks

### 5.1 Multiplicative Relationships

While additive relationships are more common, multiplicative interactions can be powerful in certain network architectures. Consider:

```math
Y_i = 12w_1w_2
```

### 5.2 Derivation of Partial Derivative

The partial derivative with respect to w1 is:

```math
\frac{\partial Y_i}{\partial w_1} = 12 \times (w_2 \frac{\partial}{\partial w_1} w_1 + w_1 \frac{\partial}{\partial w_1} w_2) = 12w_2
```

### 5.3 Resulting Gradient Computation

Therefore, the gradient of the cost function with respect to w1 is:

```math
\frac{\partial C}{\partial w_1} = \frac{\partial C}{\partial Y_i} \times 12w_2
```

### 5.4 Analysis of Multiplicative Interactions

This multiplicative interaction introduces several interesting properties:

1. Interdependence: The gradient for each weight depends on the value of the other weight, creating a more complex interdependence between parameters.

2. Potential for Stronger Interactions: Multiplicative interactions can lead to stronger feature interactions, potentially capturing more complex patterns in the data.

3. Challenges in Training: The interdependence can make training more difficult, as changes in one weight can have amplified effects on the gradients of other weights_matrix.

4. Sparsity Promotion: In some cases, multiplicative interactions can promote sparsity in the network, as small values in one weight can effectively "gate" the contribution of the other.

## 6. Conclusion and Broader Implications

The derivations presented here form the mathematical backbone of gradient-based learning in neural networks. They illuminate several key principles:

1. Layered Computation: The chain rule allows for efficient, layer-wise computation of gradients, forming the basis of the backpropagation algorithm.

2. Local and Global Information: Each weight update incorporates both local information (the neuron's activation and local derivatives) and global information (the propagated error gradient).

3. Non-linearity is Key: The introduction of non-linear activation functions is crucial for the expressive power of neural networks, allowing them to approximate complex functions.

4. Architectural Choices Matter: Different types of neuronal interactions (linear, non-linear, multiplicative) lead to different gradient computations, affecting how the network learns and what kinds of patterns it can efficiently represent.

5. Balancing Act: The interplay between forward propagation (determining activations) and backward propagation (computing gradients) creates a complex dynamical system, where the choice of architecture, activation functions, and initialization can greatly impact learning dynamics.

Understanding these mathematical principles is crucial not only for implementing neural networks but also for advancing the field. It provides insights into:
- Designing new network architectures
- Developing more efficient learning algorithms
- Understanding and mitigating issues like vanishing or exploding gradients
- Interpreting and visualizing the internal representations learned by deep networks

As the field of deep learning continues to evolve, these fundamental mathematical insights serve as a compass, guiding the development of more powerful, efficient, and interpretable neural network models.
---



