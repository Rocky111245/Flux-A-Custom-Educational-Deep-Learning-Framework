cmake_minimum_required(VERSION 3.26)
project(FLUX)

set(CMAKE_CXX_STANDARD 23)

include_directories(${CMAKE_SOURCE_DIR})  # Root directory

# Define source files grouped by directory
set(LOSS_FUNCTIONS_SRC
        "Loss_Functions/Loss_Functions.cpp"
)

set(NEURAL_BLOCKS_SRC
        "Multi Layer Perceptron/Neural_Blocks/Neural_Blocks.cpp"
)

set(NEURAL_LAYERS_SRC
        "Multi Layer Perceptron/Neural_Layers/Neural_Layer_Skeleton.cpp"
)

set(OPTIMIZATION_ALGORITHMS_SRC
        "Multi Layer Perceptron/Optimization_Algorithms/Train_Block_by_Backpropagation.cpp"
)

set(UTILITY_FUNCTIONS_SRC
        "Utility_Functions/Utility_Functions.cpp")

set(MATRIX_DIMENSION_CHECKS_SRC
        "Matrix_Dimension_Checks/Matrix_Assertion_Checks.cpp")

# WebAssembly source files
set(WASM_SOURCES
        "Web_Assembly_Modules/src/wasm_exports.cpp"
        "Web_Assembly_Modules/src/wasm_interface.cpp"
        "Transformer_Architecture/Token_Embedding/Token_Embedding.cpp"
        "Transformer_Architecture/Token_Embedding/Token_Embedding.h"
        "Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.cpp"
        "Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.h"
        "Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.cpp"
        "Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.h"
        "Transformer_Architecture/Embedding_Operations/Embedding_Operations.cpp"
        "Transformer_Architecture/Embedding_Operations/Embedding_Operations.h"
        Transformer_Architecture/Tokenizer/Tokenizer.cpp
        Transformer_Architecture/Tokenizer/Tokenizer.h
        Transformer_Architecture/Test_Data/Test_Data_Loading_Library.cpp
        Transformer_Architecture/Test_Data/Test_Data_Loading_Library.h
        Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.cpp
        Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.h
        Transformer_Architecture/Attention_Block/Multi_Head_Attention.cpp
        Transformer_Architecture/Attention_Block/Multi_Head_Attention.h
)

# Detect if we're building with Emscripten
if(EMSCRIPTEN)
    message(STATUS "Building with Emscripten for WebAssembly")

    # (A) Create an Emscripten "MatrixLibrary" from source,
    add_library(MatrixLibrary STATIC
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/MatrixLibrary.cpp"
    )


    target_include_directories(MatrixLibrary PUBLIC
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks"
    )


    # (B) Create FLUX_wasm executable,
    #     adding the sources from the project:

    add_executable(${PROJECT_NAME}_wasm
            ${LOSS_FUNCTIONS_SRC}
            ${NEURAL_BLOCKS_SRC}
            ${NEURAL_LAYERS_SRC}
            ${OPTIMIZATION_ALGORITHMS_SRC}
            ${MATRIX_DIMENSION_CHECKS_SRC}
            ${UTILITY_FUNCTIONS_SRC}
            ${WASM_SOURCES}
    )

    # (C) Link the Emscripten-compiled MatrixLibrary into FLUX_wasm
    target_link_libraries(${PROJECT_NAME}_wasm PRIVATE MatrixLibrary)

    #
    # (D) The usual Emscripten settings for generating .wasm + .js:
    #
    set(EMSCRIPTEN_LINK_FLAGS
            "-s WASM=1 \
 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
 -s ALLOW_MEMORY_GROWTH=1 \
 -s MODULARIZE=1 \
 -s EXPORT_NAME=createNeuralNetworkModule \
 -s EXPORTED_FUNCTIONS=['_createCustomBlock','_getBlockSize','_createMultiLayerNetwork','_getNetworkLoss','_malloc','_free','_getPredictions','_getInputPatterns','_getTargetValues','_Initialize_Trainer_For_One_Step_Iteration','_Train_Network_By_One_Iteration','_Get_Block_Loss']"
    )

    set_target_properties(${PROJECT_NAME}_wasm PROPERTIES LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS}")

    # (E) Copy .js + .wasm to the Web_Visualizations folder:
    add_custom_command(TARGET ${PROJECT_NAME}_wasm POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/Web_Visualizations
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_wasm.js
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_wasm.wasm
            ${CMAKE_SOURCE_DIR}/Web_Visualizations/
    )

else()
    #
    # Native build
    #
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
            "Transformer_Architecture/Token_Embedding/Token_Embedding.cpp"
            "Transformer_Architecture/Token_Embedding/Token_Embedding.h"
            "Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.cpp"
            "Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.h"
            "Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.cpp"
            "Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.h"
            "Transformer_Architecture/Embedding_Operations/Embedding_Operations.cpp"
            "Transformer_Architecture/Embedding_Operations/Embedding_Operations.h"
            Transformer_Architecture/Tokenizer/Tokenizer.cpp
            Transformer_Architecture/Tokenizer/Tokenizer.h
    )
    target_include_directories(${PROJECT_NAME} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks")
    target_link_directories(${PROJECT_NAME} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug")
    target_link_libraries(${PROJECT_NAME} PRIVATE libMatrixLibrary)

    # Copy the Matrix Library DLL to the output directory (native only)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )



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
            Tensor_Library/Tensor_Library.h
            Convolutional_Neural_Networks/Convolutional_Layer.cpp
            Convolutional_Neural_Networks/Convolutional_Layer.h
            "Convolutional_Neural_Networks/Padding Functions/Tensor_Padding.cpp"
            "Convolutional_Neural_Networks/Padding Functions/Tensor_Padding.h"
            Convolutional_Neural_Networks/Pooling_Layers.cpp
            Convolutional_Neural_Networks/Pooling_Layers.h
    )

    target_include_directories(CNN_main PRIVATE
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks"
            "${CMAKE_SOURCE_DIR}/Convolutional_Neural_Networks/Matrix_Migration_Functions"
    )

    target_link_directories(CNN_main PRIVATE
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug"
    )

    target_link_libraries(CNN_main PRIVATE libMatrixLibrary)

    add_custom_command(TARGET CNN_main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
            $<TARGET_FILE_DIR:CNN_main>
    )


    add_executable(Transformer_main
            Transformer_Architecture/main.cpp
            Transformer_Architecture/Tokenizer/Tokenizer.cpp
            Transformer_Architecture/Token_Embedding/Token_Embedding.cpp
            Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.cpp
            Transformer_Architecture/Positional_Embeddings/Learned_Positional_Embeddings.cpp
            Transformer_Architecture/Embedding_Operations/Embedding_Operations.cpp
            ${LOSS_FUNCTIONS_SRC}
            ${NEURAL_BLOCKS_SRC}
            ${NEURAL_LAYERS_SRC}
            ${OPTIMIZATION_ALGORITHMS_SRC}
            ${MATRIX_DIMENSION_CHECKS_SRC}
            ${UTILITY_FUNCTIONS_SRC}
            Tensor_Library/Tensor_Library.cpp
            Transformer_Architecture/main.cpp
            Transformer_Architecture/Test_Data/Test_Data_Loading_Library.cpp
            Transformer_Architecture/Test_Data/Test_Data_Loading_Library.h
            Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.cpp
            Transformer_Architecture/Single_Attention_Head/Single_Attention_Head.h
            Transformer_Architecture/Attention_Block/Multi_Head_Attention.cpp
            Transformer_Architecture/Attention_Block/Multi_Head_Attention.h

    )

    target_include_directories(Transformer_main PRIVATE
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks"
            "${CMAKE_SOURCE_DIR}/Transformer_Architecture"
    )

    target_link_directories(Transformer_main PRIVATE
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug"
    )

    target_link_libraries(Transformer_main PRIVATE libMatrixLibrary)

    add_custom_command(TARGET Transformer_main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
            $<TARGET_FILE_DIR:Transformer_main>
    )

endif()
