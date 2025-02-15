cmake_minimum_required(VERSION 3.26)
project(_Discriminative_Dense_Neural_Network_Framework)

set(CMAKE_CXX_STANDARD 23)

include_directories(${CMAKE_SOURCE_DIR})  # Add the root directory to header search paths

# Main Neural Network executable (without Tokenizer.cpp)
add_executable(_Discriminative_Dense_Neural_Network_Framework
        main.cpp
        "Neural Network Framework.cpp"
        "Neural Network Framework.h"
        "MSEGraphPlotter.cpp"
        "MSEGraphPlotter.h"
        "NNStructureVis .cpp"
        "NNStructureVis .h"
        "Loss Functions/Loss_Functions.cpp"
        "Loss Functions/Loss_Functions.h"
        "Forward Pass/Forward_Pass.cpp"
        "Forward Pass/Forward_Pass.h"
        "Neural_Blocks/Neural_Blocks.cpp"
        "Neural_Blocks/Neural_Blocks.h"
        "Initialize Weights/Initialize_Weights.cpp"
        "Initialize Weights/Initialize_Weights.h"
        "Neural_Layers/Neural_Layer_Skeleton.cpp"  # Corrected path
        "Neural_Layers/Neural_Layer_Skeleton.h"    # Corrected path
)

# Tokenizer executable using the existing Tokenizer.cpp with main()
add_executable(Tokenizer
        "NLP Utilities/Tokenizer.cpp"

)

# Common settings for both executables
foreach(TARGET _Discriminative_Dense_Neural_Network_Framework Tokenizer)
    target_include_directories(${TARGET} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks")
    target_link_directories(${TARGET} PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug")
    target_link_libraries(${TARGET} PRIVATE libMatrixLibrary)

    add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
            $<TARGET_FILE_DIR:${TARGET}>
    )
endforeach()