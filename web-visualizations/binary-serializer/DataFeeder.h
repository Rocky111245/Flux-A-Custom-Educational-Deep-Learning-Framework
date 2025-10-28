//
// Created by Rocky170 on 10/25/2025.
//

#ifndef DATA_FEEDER_H
#define DATA_FEEDER_H



class TensorDataExtractor;


//This class's object lifetime is NOT directly tied to the lifetime of the TensorDataExtractor object it is used with.
//It creates its own copy of the tensor binary-serializer and stores it in a binary file.
//The binary file is stored in the Data folder.

class DataFeeder {
public:
    DataFeeder() noexcept;           // keep as-is if your ctor truly never throws

    // Serialize a full tensor + metadata
    void Serialize(const TensorDataExtractor& tensor);
};


#endif //DATA_FEEDER_H
