cmake_minimum_required(VERSION 3.26)
project(FluxNeuralNetworkFramework)
set(CMAKE_CXX_STANDARD 23)
include_directories(${CMAKE_SOURCE_DIR})


# ============================================
# SHARED COMPONENT INTERNAL LIBRARIES
# ============================================

add_library(NeuralCore STATIC
        activation-function-derivatives/ActivationFunctionDerivatives.cpp
        activation-functions/ActivationFunctions.cpp
        loss-functions/TensorLossFunctions.cpp
        tensor-library/TensorLibrary.cpp
        matrix-library/MatrixLibrary.cpp
)

# ============================================
# FEED FORWARD MODEL
# ============================================

add_library(MultiLayerPerceptron STATIC
        model-architectures/multi-layer-perceptron/mlp-feed-forward/NeuralLayer.cpp
        model-architectures/multi-layer-perceptron/mlp-feed-forward/NeuralBlock.cpp
)
target_link_libraries(MultiLayerPerceptron PUBLIC NeuralCore)

target_include_directories(MultiLayerPerceptron PUBLIC
        ${CMAKE_SOURCE_DIR}/model-architectures/multi-layer-perceptron/mlp-feed-forward
)


# ============================================
# TRANSFORMER MODEL
# ============================================

add_library(TransformerModel STATIC
        model-architectures/transformers/token_embedding/TokenEmbedding.cpp
        model-architectures/transformers/positional-embeddings/FixedPositionalEncoding.cpp
        model-architectures/transformers/embedding-operations/EmbeddingOperations.cpp
        model-architectures/transformers/tokenizer/Tokenizer.cpp
        model-architectures/transformers/single-attention-head/SingleAttentionHead.cpp
        model-architectures/transformers/multi-head-attention/MultiHeadAttention.cpp
        model-architectures/transformers/residual-stream/ResidualStream.cpp
        model-architectures/transformers/layer-normalization/LayerNormalization.cpp
        model-architectures/transformers/batch-operations/BatchOperations.cpp
)
target_link_libraries(TransformerModel PUBLIC NeuralCore PUBLIC MultiLayerPerceptron)
target_include_directories(TransformerModel PUBLIC
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/batch-operations
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/embedding-operations
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/layer-normalization
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/multi-head-attention
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/positional-embeddings
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/residual-stream
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/single-attention-head
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/token_embedding
        ${CMAKE_SOURCE_DIR}/model-architectures/transformers/tokenizer
)


# ============================================
# UTILITY LIBRARIES
# ============================================
add_library(WebVisualization STATIC
        web-visualizations/data-loader/tensor-data-extractor/TensorDataExtractor.cpp
        web-visualizations/data-loader/binary-serializer/DataFeeder.cpp
        web-visualizations/data-loader/DataOrchestrator.cpp
)
target_link_libraries(WebVisualization PRIVATE NeuralCore PRIVATE TransformerModel)
target_include_directories(WebVisualization PUBLIC
        ${CMAKE_SOURCE_DIR}/web-visualizations/data-loader
)


# ============================================
# EXECUTABLES
# ============================================

add_executable(${PROJECT_NAME} main.cpp
        web-visualizations/data-loader/DataOrchestrator.cpp
        web-visualizations/data-loader/DataOrchestrator.h)
target_link_libraries(${PROJECT_NAME} PRIVATE NeuralCore PRIVATE TransformerModel PRIVATE MultiLayerPerceptron)

add_executable(Transformer_main
        model-architectures/transformers/main.cpp
)
target_link_libraries(Transformer_main PRIVATE TransformerModel)


add_executable(MLP_main
        model-architectures/multi-layer-perceptron/main.cpp
)
target_link_libraries(MLP_main PRIVATE MultiLayerPerceptron)


add_executable(WebVis_main
        web-visualizations/main.cpp
)
target_link_libraries(WebVis_main PRIVATE TransformerModel )

