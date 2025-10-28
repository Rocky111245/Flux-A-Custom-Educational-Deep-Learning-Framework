//
// Created by Rocky170 on 10/25/2025.
//


//The purpose of this file is to collect all relevant tensor binary-serializer and serialize into binary format

#include "../DataFeeder.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

#include "../TensorDataExtractor.h"


void Serialize(const TensorDataExtractor& tensor) {
    if (!tensor.is_initialized()) {
        throw std::invalid_argument("TensorDataExtractor is not initialized. Error detected in Data_Feeder::Serialize");
    }

    const uint8_t tensor_type = tensor.tensor_type();      // 8 bits
    const uint32_t layer_number = tensor.layer_number();   // 32 bits
    const uint32_t rows = tensor.rows();                   // 32 bits
    const uint32_t columns = tensor.columns();             // 32 bits
    const uint32_t depth = tensor.depth();                 // 32 bits
    const uint32_t num_elements = tensor.size();           // 32 bits
    const float* data = tensor.get_ptr();                  // not determined at compile time.

    const uint32_t float_data_bytes = num_elements * sizeof(float);


    // Total size of tensor binary-serializer (excluding this size field itself)
    // To skip this tensor: read this value, then seek forward by this many bytes
    const uint32_t total_tensor_size =sizeof(tensor_type) +sizeof(layer_number) +sizeof(rows) +sizeof(columns) +sizeof(depth) +float_data_bytes; // 32 bits



    std::ofstream out("./binary-serializer/binary-serializer.bin", std::ios::binary | std::ios::app);
    if (!out) {
        std::cerr << "Error opening file: " << strerror(errno) << "\n";
        return;
    }

    // Total size header (useful for knowing the size of  a tensor since they are stored contiguously)
    out.write(reinterpret_cast<const char*>(&total_tensor_size), sizeof(total_tensor_size));
    // Metadata
    out.write(reinterpret_cast<const char*>(&tensor_type), sizeof(tensor_type));
    out.write(reinterpret_cast<const char*>(&layer_number), sizeof(layer_number));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));

    // Data
    out.write(reinterpret_cast<const char*>(data), float_data_bytes);

    out.close();
}


//Clear Binary file
void Clear_Binary_File() {

    // Step 1: Check if file exists
    if (std::filesystem::exists("./binary-serializer/binary-serializer.bin")) {
        std::cout << "File exists: " << "./binary-serializer/binary-serializer.bin"<< std::endl;

        // Step 2: Clear the file by opening in truncation mode
        if (std::ofstream ofs("./binary-serializer/binary-serializer.bin", std::ios::binary | std::ios::trunc); ofs) {
            std::cout << "File cleared successfully." << std::endl;
        } else {
            std::cerr << "Failed to open the file for truncation." << std::endl;
        }
    } else {
        std::cout << "File does not exist: " << "./binary-serializer/binary-serializer.bin" << std::endl;
    }
}



/*  Layout--> |TENSOR TYPE| |LAYER NUMBER| |ROWS| |COLUMNS| |DEPTH| | B  I  G   D A T A |   */







