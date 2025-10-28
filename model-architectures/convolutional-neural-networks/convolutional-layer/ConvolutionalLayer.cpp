#include "ConvolutionalLayer.h"

#include "../utility-migratory-functions/UtilityMigratoryFunctions.h"
#include "tensor-library/TensorLibrary.h"
#include "../kernels/Kernels.h"



Convolutional_Layer_Skeleton::Convolutional_Layer_Skeleton(Tensor& input_tensor, int number_of_kernels, ConvActivationType activation_type,
                                                           PaddingStrategy padding_strategy , int stride):


{



}



void Convolutional_Layer_Skeleton::Initialize_All_Tensors(){

    int kernel_size=kernels_vectors_[0].size();
    int kernel_depth=kernels_vectors_[0].depth();
    int padding_size = Padding_Get_Size(input_tensor_.rows(), kernel_size, stride_, padding_strategy_);

    //reserve space in the kernel_ vector according to the number of kernels in the layer
    kernels_vectors_.reserve(number_of_kernels_);

    //This function generates proper padding in the padded_input_tensor. It also auto-resizes the padded_input_tensor to the proper size.
    Padding_Apply_Strategy_To_Tensor(padded_input_tensor_, input_tensor_, kernel_size, stride_, padding_strategy_);




    int im2rows_rows = Padding_Get_Output_Columns(input_tensor_.rows(), kernel_size, stride_, padding_size);
    int im2rows_columns = Padding_Get_Output_Columns(input_tensor_.columns(), kernel_size, stride_, padding_size);

    //this tensor gets the im2rows binary-serializer for all channels
    im2rows_tensor_(im2rows_rows, im2rows_columns,kernel_depth);

    //Usually memory allocation is kept separate from the convolutional operation.However, due to the complexity of the
    //implementation, we decided to do still resize it in this function for safety.

    //This function extracts the im2rows tensor from the padded_input_tensor. Auto-resizes the im2rows_tensor to the proper size.
    Extract_Patches_Im2rows(im2rows_tensor_,padded_input_tensor_,kernel_size,padding_size,stride_);

    //loop over and flatten all kernels in the vector
    for(int i=0;i<number_of_kernels_;i++){
        flattened_kernels_vectors_.push_back(kernels_vectors_[i].Flatten_Kernel());
    }

}

int Convolutional_Layer_Skeleton::Get_Output_Tensor_Columns() const {
    return (input_tensor_.rows()+2())
}