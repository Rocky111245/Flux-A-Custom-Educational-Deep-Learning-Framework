#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_PADDING_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_PADDING_H

#include "Tensor_Library/Tensor_Library.h"

/*-------------------------- Padding Strategy Enum -------------------------*/
enum PaddingStrategy {
    VALID,  /* No padding - output will be smaller than input */
    SAME    /* Padding to maintain input dimensions (when stride=1)  Maintain same output size */
};

int Get_Output_Rows(int input_rows,int kernel_size, int stride_size, PaddingStrategy strategy);
int Get_Output_Columns(int input_columns, int kernel_size, int stride_size, PaddingStrategy strategy);

void Tensor_Apply_Padding_By_Strategy(Tensor &padded_tensor,const Tensor &input_tensor,int kernel_size,int stride,PaddingStrategy strategy);


#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TENSOR_PADDING_H