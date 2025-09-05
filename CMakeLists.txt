cmake_minimum_required(VERSION 3.26)
project(FLUX)

set(CMAKE_CXX_STANDARD 23)

include_directories(${CMAKE_SOURCE_DIR})  # Root directory

# Matrix Library (local folder version)
add_library(MatrixLibrary STATIC
        Cpp-Matrix-Library-for-Neural-Networks/MatrixLibrary.cpp
)
target_include_directories(MatrixLibrary PUBLIC
        ${CMAKE_SOURCE_DIR}/Cpp-Matrix-Library-for-Neural-Networks
)

# Define grouped source files
set(LOSS_FUNCTIONS_SRC
        Loss_Functions/Loss_Functions.cpp
        Loss_Functions/Tensor_Loss_Functions.cpp

)

set(NEURAL_BLOCKS_SRC
        Multi\ Layer\ Perceptron/Neural_Blocks/Neural_Blocks.cpp
)

set(NEURAL_LAYERS_SRC
        Multi\ Layer\ Perceptron/Neural_Layers/Neural_Layer_Skeleton.cpp
)

set(OPTIMIZATION_ALGORITHMS_SRC
        Multi\ Layer\ Perceptron/Optimization_Algorithms/Train_Block_by_Backpropagation.cpp
)

set(UTILITY_FUNCTIONS_SRC
        Utility_Functions/Utility_Functions.cpp
)

set(MATRIX_DIMENSION_CHECKS_SRC
        Matrix_Dimension_Checks/Matrix_Assertion_Checks.cpp
)

# Tokenizer/Embedding components (shared between main & transformer)
set(TRANSFORMER_COMPONENTS
        Transformer_Architecture/Token_Embedding/Token_Embedding.cpp
        Transformer_Architecture/Token_Embedding/Token_Embedding.h
        Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.cpp
        Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.h
        Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.cpp
        Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.h
        Transformer_Architecture/Embedding_Operations/Embedding_Operations.cpp
        Transformer_Architecture/Embedding_Operations/Embedding_Operations.h
        Transformer_Architecture/Tokenizer/Tokenizer.cpp
        Transformer_Architecture/Tokenizer/Tokenizer.h
)

# --- Main Executable ---
add_executable(${PROJECT_NAME}
        main.cpp
        ${LOSS_FUNCTIONS_SRC}
        ${NEURAL_BLOCKS_SRC}
        ${NEURAL_LAYERS_SRC}
        ${OPTIMIZATION_ALGORITHMS_SRC}
        ${MATRIX_DIMENSION_CHECKS_SRC}
        ${UTILITY_FUNCTIONS_SRC}
        Tensor_Library/Tensor_Library.cpp
        Tensor_Library/Tensor_Library.h
        ${TRANSFORMER_COMPONENTS}
        Activation_Function_Derivatives/Activation_Function_Derivatives.cpp
        Activation_Function_Derivatives/Activation_Function_Derivatives.h
        Activation_Functions/Activation_Functions.cpp
        Activation_Functions/Activation_Functions.h
        Loss_Functions/Tensor_Loss_Functions.cpp
        Loss_Functions/Tensor_Loss_Functions.h
)
target_link_libraries(${PROJECT_NAME} PRIVATE MatrixLibrary)

# --- CNN Executable ---
add_executable(CNN_main
        Convolutional_Neural_Networks/main.cpp
        Convolutional_Neural_Networks/Utility_Migratory_Functions/Utility_Migratory_Functions.cpp
        ${LOSS_FUNCTIONS_SRC}
        ${NEURAL_BLOCKS_SRC}
        ${NEURAL_LAYERS_SRC}
        ${OPTIMIZATION_ALGORITHMS_SRC}
        ${MATRIX_DIMENSION_CHECKS_SRC}
        ${UTILITY_FUNCTIONS_SRC}
        Convolutional_Neural_Networks/Kernels/Kernels.cpp
        Convolutional_Neural_Networks/Kernels/Kernels.h
        Tensor_Library/Tensor_Library.cpp
        Convolutional_Neural_Networks/Convolutional_Layer.cpp
        Convolutional_Neural_Networks/Convolutional_Layer.h
        Convolutional_Neural_Networks/Padding\ Functions/Tensor_Padding.cpp
        Convolutional_Neural_Networks/Padding\ Functions/Tensor_Padding.h
        Convolutional_Neural_Networks/Pooling_Layers.cpp
        Convolutional_Neural_Networks/Pooling_Layers.h
)
target_link_libraries(CNN_main PRIVATE MatrixLibrary)

# --- Transformer Executable ---
add_executable(Transformer_main
        Transformer_Architecture/main.cpp
        ${TRANSFORMER_COMPONENTS}
        ${LOSS_FUNCTIONS_SRC}
        ${NEURAL_BLOCKS_SRC}
        ${NEURAL_LAYERS_SRC}
        ${OPTIMIZATION_ALGORITHMS_SRC}
        ${MATRIX_DIMENSION_CHECKS_SRC}
        ${UTILITY_FUNCTIONS_SRC}
        Tensor_Library/Tensor_Library.cpp
        Activation_Function_Derivatives/Activation_Function_Derivatives.cpp
        Activation_Functions/Activation_Functions.cpp
        Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.cpp
        Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.h
        Transformer_Architecture/Attention_Block/Multi_Head_Attention.cpp
        Transformer_Architecture/Attention_Block/Multi_Head_Attention.h
        Transformer_Architecture/Residual_and_Normalization/Residual_Stream.cpp
        Transformer_Architecture/Residual_and_Normalization/Residual_Stream.h
        Transformer_Architecture/Residual_and_Normalization/Layer_Normalization.cpp
        Transformer_Architecture/Residual_and_Normalization/Layer_Normalization.h
        Transformer_Architecture/MLP_Layers_For_Transformer/Neural_Layer.cpp
        Transformer_Architecture/MLP_Layers_For_Transformer/Neural_Layer.h
        Transformer_Architecture/MLP_Layers_For_Transformer/Neural_Block.cpp
        Transformer_Architecture/MLP_Layers_For_Transformer/Neural_Block.h
        Transformer_Architecture/Batch_Operations/Batch_Operations.cpp
        Transformer_Architecture/Batch_Operations/Batch_Operations.h
)
target_link_libraries(Transformer_main PRIVATE MatrixLibrary)
