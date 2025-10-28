//
// Created by Rocky170 on 10/23/2025.
//

#ifndef TENSOR_EXTRACTION_OPERATIONS_H
#define TENSOR_EXTRACTION_OPERATIONS_H
class Tensor;

#include <cstdint>
#include <vector>



enum class TENSOR_TYPE : uint8_t {
  MLP_INPUT,MLP_PRE_ACTIVATION,MLP_POST_ACTIVATION,MLP_OUTPUT
};



// Container for tensor binary-serializer prepared for WebGPU/WebAssembly transmission.
// Supports single initialization and subsequent read-only access.
class TensorDataExtractor {
public:
    TensorDataExtractor() = default;

    // Initialize tensor binary-serializer. Can only be called once per object.
    void Set_Data(const Tensor& tensor, int layer_number,TENSOR_TYPE tensor_type);

    // Pointer for WebGPU/WASM interop
    float* get_ptr();
    const float* get_ptr() const;

    uint32_t size() const;
    uint32_t rows() const;
    uint32_t columns() const;
    uint32_t depth() const;
    uint32_t layer_number() const;

    uint8_t tensor_type() const;

    bool is_initialized() const noexcept;

private:
    std::vector<float> data_;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t depth_ = 0;
    uint32_t layer_number_ = 0;
    uint8_t tensor_type_ = 0;
};



#endif //TENSOR_EXTRACTION_OPERATIONS_H
