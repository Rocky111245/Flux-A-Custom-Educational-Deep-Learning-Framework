//
// Created by Rocky170 on 10/23/2025.
//

#include "tensor-library/TensorLibrary.h"
#include "../TensorDataExtractor.h"
#include <cstdint>
#include <stdexcept>

// Container for tensor data prepared for WebGPU/WebAssembly transmission.
// Supports single initialization and subsequent read-only access.
void TensorDataExtractor::Set_Data(const Tensor& tensor, const int layer_number,TENSOR_TYPE tensor_type) {
    // Check if already initialized
    if (!data_.empty()) {
        throw std::logic_error("TensorDataExtractor already initialized");
    }
    // Validate layer number
    if (layer_number < 0) {
        throw std::invalid_argument(
            "Layer number cannot be negative: " + std::to_string(layer_number)
        );
    }

    rows_ = static_cast<uint32_t>(tensor.rows());
    columns_ = static_cast<uint32_t>(tensor.columns());
    depth_ = static_cast<uint32_t>(tensor.depth());
    layer_number_ = static_cast<uint32_t>(layer_number);
    tensor_type_=static_cast<uint8_t>(tensor_type);


    // Check for size overflow using 64-bit intermediate
    const uint64_t size_check =
        uint64_t{rows_} * uint64_t{columns_} * uint64_t{depth_};
    if (size_check > UINT32_MAX) {
        throw std::overflow_error(
            "Tensor size exceeds uint32_t maximum: " + std::to_string(size_check)
        );
    }

    const uint32_t size = static_cast<uint32_t>(size_check);
    data_.resize(size);
    std::copy_n(tensor.Get_Data(), size, data_.data());
}

float* TensorDataExtractor::get_ptr() {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return data_.data();
}

const float* TensorDataExtractor::get_ptr() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return data_.data();
}

uint32_t TensorDataExtractor::size() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return static_cast<uint32_t>(data_.size());
}

uint32_t TensorDataExtractor::rows() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return rows_;
}

uint32_t TensorDataExtractor::columns() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return columns_;
}

uint32_t TensorDataExtractor::depth() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return depth_;
}

uint32_t TensorDataExtractor::layer_number() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return layer_number_;
}

uint8_t TensorDataExtractor::tensor_type() const {
    if (data_.empty()) {
        throw std::logic_error("TensorDataExtractor not initialized");
    }
    return tensor_type_;
}

bool TensorDataExtractor::is_initialized() const noexcept {
    return !data_.empty();
}
