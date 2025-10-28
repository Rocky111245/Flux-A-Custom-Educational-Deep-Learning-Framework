//
// Created by Rocky170 on 7/23/2025.
//

#include "NeuralBlock.h"
#include <initializer_list>
#include <iostream>
#include <utility>
#include "tensor-library/TensorLibrary.h"



NeuralBlock::NeuralBlock(Tensor residual_stream_input, const std::initializer_list<Neural> layer_list, const std::initializer_list<Tensor_Loss_Function> loss_function_list) :
layers_(layer_list),input_tensor_(std::move(residual_stream_input)),block_size_(layer_list.size()),loss_functions_(loss_function_list),
block_constructed(false),forward_pass_called_(false),back_propagation_done_(false),upstream_error_set_(false),downstream_error_set_(false),update_block_called_(false),
local_cache_cleared_(false),block_cache_cleared_(false)
{
    assert(layer_list.size() > 0);
    Construct_Block_Layers();
}

//This function is called once when the block is created. It will initialize all the Neural layers inside the block and set the
//inputs for every layer
void NeuralBlock::Construct_Block_Layers() {
    const size_t L = layers_.size();
    // Layer 0 shapes from block input
    layers_[0].Resize_Tensors(input_tensor_); //we just need the shape of the input
    layers_[0].Assert_Invariants(); // checks
    layers_[0].Initialize_Weights();
    layers_[0].Initialize_Biases();

    // Subsequent layers: shape from previous layer's output shape
    for (size_t i = 1; i < L; ++i) {
        const Tensor& prev_out_shape = layers_[i-1].Get_Output_View(); // allocated by Resize_Tensors
        layers_[i].Resize_Tensors(prev_out_shape);
        layers_[i].Assert_Invariants();
        layers_[i].Initialize_Weights();
        layers_[i].Initialize_Biases();
    }
    block_constructed=true;
}


//Step 1: Forward pass.
void NeuralBlock::Forward_Pass() {
    assert(!layers_.empty() && "Block must have at least one layer in NeuralBlock::Forward_Pass().");
    // If the external input shape might change between runs, recheck layer-0 shapes:


    // Run layer 0
    layers_[0].Set_Input(input_tensor_); //First input tensor comes from the block
    layers_[0].Forward_Pass();

    // Chain through the rest
    for (size_t i = 1; i < layers_.size(); ++i) {
        const Tensor& previous_output = layers_[i-1].Get_Output_View(); // [S, N_prev, B]
        // Ensure shape match;
        const Tensor& current_input = layers_[i].Get_Input_View();
        assert(current_input.rows() == previous_output.rows() && current_input.columns() == previous_output.columns() &&
       current_input.depth() == previous_output.depth() &&
       "Input shape mismatch between layers");
        layers_[i].Set_Input(previous_output);
        layers_[i].Forward_Pass();
    }

    output_tensor_ = layers_.back().Get_Output_View();
    forward_pass_called_ = true;
}


//Step 2: Backward pass
void NeuralBlock::BackPropagate(const Tensor& upstream_error_dL_da) {
    assert(block_size_>0 && "Block size must be greater than 0 in NeuralBlock::BackPropagate().");
    assert(!layers_.empty());
    assert(forward_pass_called_ && "BackPropagate() requires Forward_Pass() first");
    // Shape checks
    {
        const auto& top_out = layers_.back().Get_Output_View();
        assert(upstream_error_dL_da.rows()    == top_out.rows() &&
               upstream_error_dL_da.columns() == top_out.columns() &&
               upstream_error_dL_da.depth()   == top_out.depth() &&
               "Upstream gradient must match top layer output [S,N_last,B]");
    }
    upstream_error_entering_block_ = upstream_error_dL_da;

    //We start from the top of the stack (layers of the MLP)
    for (size_t i = layers_.size(); i-- > 0; ) {
        if (i == layers_.size() - 1) {
            layers_[i].Backpropagate(upstream_error_dL_da);
        } else {
            const Tensor& error_gradient = layers_[i + 1].Get_Downstream_Error(); // [S, I_i, B]
            layers_[i].Backpropagate(error_gradient);
        }
    }

    downstream_error_leaving_block_ = layers_.front().Get_Downstream_Error(); // to previous block
    back_propagation_done_ = true;
    upstream_error_set_= true;
    downstream_error_set_  = true;
}


void NeuralBlock::Perform_Checks() const {
    assert(!layers_.empty());
    assert(block_size_ == layers_.size() && "block_size_ out of sync in NeuralBlock::Perform_Checks()");
    assert(input_tensor_.rows() > 0 && input_tensor_.columns() > 0 && input_tensor_.depth() > 0);
    // Chain shape consistency:
    for (size_t i = 0; i + 1 < layers_.size(); ++i) {
        const auto& a = layers_[i].Get_Output_View();
        const auto& b = layers_[i+1].Get_Input_View();
        assert(a.rows()    == b.rows()    && "Sequence length mismatch between layers in NeuralBlock::Perform_Checks()");
        assert(a.depth()   == b.depth()   && "Batch size mismatch between layers NeuralBlock::Perform_Checks()");
        assert(a.columns() == b.columns() && "Feature mismatch (I_next must equal N_prev) NeuralBlock::Perform_Checks()");
    }
}

//  Step 3: Update parameters after a full forward and backward pass.
//  This is usually after the full network has been backpropagated.
void NeuralBlock::Update_Block(const float learning_rate) {
    assert(back_propagation_done_ && "Update_Block() requires BackPropagate() first");
    for (auto& layer : layers_) {
        layer.Update_Parameters(learning_rate);
    }
    update_block_called_ = true;
}



//this triggers the cache clearing of the layers inside the block
void NeuralBlock::Clear_Local_Cache() {
    assert(update_block_called_ && "Clear_Local_Cache() should be called after Update_Block()");
    for (auto& layer : layers_) layer.Clear_Local_Cache();
    local_cache_cleared_ = true;
}

void NeuralBlock::Clear_Block_Cache() {
    input_tensor_.Fill(0.0f);
    output_tensor_.Fill(0.0f);
    upstream_error_entering_block_.Fill(0.0f);
    downstream_error_leaving_block_.Fill(0.0f);
    block_cache_cleared_=true;
}


//this will reset the block(cache clear+reset). HIT THIS AFTER UPDATE BLOCK IS CALLED.
void NeuralBlock::Reset_Block() {
    //clear cache
    Clear_Local_Cache();
    Clear_Block_Cache();
    //reset
    block_constructed=true;
    forward_pass_called_=false;
    back_propagation_done_=false;
    downstream_error_set_=false;
    update_block_called_=false;
    local_cache_cleared_=false;
    block_cache_cleared_=false;
    upstream_error_set_=false;
}



//Useful Getters

Tensor NeuralBlock::Get_Input_Clone() const {
    return input_tensor_;
}

const Tensor& NeuralBlock::Get_Input_View() const {
    return input_tensor_;
}

Tensor& NeuralBlock::Get_Input_Mutable() {
    return input_tensor_;
}

Tensor  NeuralBlock::Get_Output_Clone() const {
    return output_tensor_;
}

const Tensor& NeuralBlock::Get_Output_View()const {
    return output_tensor_;
}

Tensor& NeuralBlock::Get_Output_Mutable() {
    return output_tensor_;
}

//Will only work if backpropagation has been called already. Read-only view for any purpose (debug etc.).
const Tensor& NeuralBlock::Get_Downstream_Error() const {
    assert(downstream_error_leaving_block_.rows()!=0 && downstream_error_leaving_block_.columns()!=0 && downstream_error_leaving_block_.depth()!=0
        && "Downstream error has not been set in NeuralBlock::Get_Downstream_Error(). Call backpropagation first.");
    assert(downstream_error_set_ && "Downstream error has not been set in NeuralBlock::Get_Downstream_Error(). "
                                   "Call backpropagation first, it will handle downstream error.");
    return downstream_error_leaving_block_;
}

//for now, we assume there is exactly one loss head. This is good enough for generative,classification and regression tasks.
//TODO

// void NeuralBlock::Attach_Loss_Heads(const Tensor& prediction) {
//     int number_of_outer_layer_neurones=layers_[layers_.size()-1].Get_Neuron_Count();
//     int number_of_loss_heads=loss_functions_.size();
//     int count=0;
//
//     //we count how many neurones were demanded to be attached to the loss head by the user
//     for (int i=0;i<number_of_loss_heads;i++) {
//         count+=loss_functions_[i].Get_Neurone_Count();
//     }
//
//     if (count!=number_of_outer_layer_neurones) {
//         throw std::logic_error("The loss heads demanded a larger number of neurones to be attached than the number of "
//                                "neurones actually present in the outer layer. Error in NeuralBlock::Attach_Loss_Heads()");
//     }
//
//     //later I will work on feature splitting vs batch splitting.Right now, for testing, we will assume that all neurones
//     //are tied to the same loss function
//
//
//     //This implementation is for one loss head
//     loss_functions_[0].Attach_Loss_Head(prediction);
// }

float NeuralBlock::Compute_Loss(const Tensor& prediction, const Tensor& target) {
    float loss=0.0f;

    if (loss_functions_.empty()) {
        throw std::logic_error("No loss functions were attached to the block. Error in NeuralBlock::Compute_Loss()");
    }
    loss=loss_functions_[0].Compute(prediction,target);
    return loss;
}

//output_= Tensor(S, N, B); // [S, N, B] -> Our output is constructed like this.


// Train the block for N iterations using a fixed input/output pair. An Example.
float NeuralBlock::Train(const Tensor& input,
                          const Tensor& target,
                          const float learning_rate,
                          const int iterations,
                          const int print_every) {
    if (iterations <= 0) {
        throw std::invalid_argument("Train: iterations must be > 0");
    }
    if (learning_rate <= 0.0f || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Train: learning rate must be finite and > 0");
    }
    if (loss_functions_.empty()) {
        throw std::logic_error("Train: block has no loss functions attached");
    }

    assert(block_constructed && "Please call the constructor first");

    float total_loss = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        // Step 1: Set input and forward
        input_tensor_ = input;
        Forward_Pass();

        // Step 2: Compute loss and get gradient
        float loss = loss_functions_[0].Compute(output_tensor_, target);
        const Tensor& dL_da = loss_functions_[0].Get_Downstream_Error_View();

        // Step 3: Backward and update
        BackPropagate(dL_da);
        Perform_Checks();
        Update_Block(learning_rate);

        // Step 4: Clear caches
        Reset_Block();

        total_loss += loss;

        // Logging checkpoint
        if ((i + 1) % print_every == 0 || i == iterations - 1) {
            std::cout << "[Iteration " << (i + 1) << "/" << iterations
                      << "] Loss = " << loss << std::endl;
        }
    }

    return total_loss / iterations; // Return average loss over N iterations
}




