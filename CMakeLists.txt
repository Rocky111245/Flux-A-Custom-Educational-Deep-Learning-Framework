cmake_minimum_required(VERSION 3.26)
project(FLUX)

set(CMAKE_CXX_STANDARD 23)

# Include directories for header files
include_directories(${CMAKE_SOURCE_DIR})  # Add the root directory to header search paths

# Define source files grouped by directory


set(LOSS_FUNCTIONS_SRC
        "Loss_Functions/Loss_Functions.cpp"
)

set(NEURAL_BLOCKS_SRC
        "Neural_Blocks/Neural_Blocks.cpp"
)

set(NEURAL_LAYERS_SRC
        "Neural_Layers/Neural_Layer_Skeleton.cpp"
)

set(OPTIMIZATION_ALGORITHMS_SRC
        "Optimization_Algorithms/Train_Block_by_Backpropagation.cpp"
)

set(UTILITY_FUNCTIONS_SRC
        "Utility_Functions/Utility_Functions.cpp")

set(MATRIX_DIMENSION_CHECKS_SRC
        "Matrix_Dimension_Checks/Matrix_Assertion_Checks.cpp")

# Main executable
add_executable(${PROJECT_NAME}
        main.cpp


        ${LOSS_FUNCTIONS_SRC}
        ${NEURAL_BLOCKS_SRC}
        ${NEURAL_LAYERS_SRC}
        ${OPTIMIZATION_ALGORITHMS_SRC}
        ${MATRIX_DIMENSION_CHECKS_SRC}
        ${UTILITY_FUNCTIONS_SRC}
)

# Matrix library dependency
target_include_directories(${PROJECT_NAME} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks")
target_link_directories(${PROJECT_NAME} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug")
target_link_libraries(${PROJECT_NAME} PRIVATE libMatrixLibrary)

# Copy the Matrix Library DLL to the output directory after build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)