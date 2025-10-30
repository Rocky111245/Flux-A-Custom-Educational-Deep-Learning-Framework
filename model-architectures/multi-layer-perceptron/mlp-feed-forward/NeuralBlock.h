//
// Created by Rocky170 on 7/23/2025.
//

// NeuralBlock: Orchestrates a stack of dense layers + loss.
//
// Purpose
// -------
//This class is a high level orchestrator responsible for dealing with the operations related to an MLP layer.This is the main interface which will
//run the forward pass,backpropagation engine of multiple neural class layers.

// High-level driver for an MLP-style block. Owns a sequence of `Neural` layers,
// manages forward and backward passes across them, and applies parameter updates and mechanistic intervention experiments.
// The block also integrates one or more loss functions (currently only one loss function has been coded).

#ifndef NEURAL_BLOCK_H
#define NEURAL_BLOCK_H
#include "NeuralLayer.h"
#include "loss-functions/TensorLossFunctions.h"


enum neurone_intervention_type {
    ABLATE,
    MEAN_ABLATE,
    PATCH,
    RANDOM,
    CLAMP,
    SCALE
};

class NeuralBlock {
public:
   explicit NeuralBlock(Tensor residual_stream_input, std::initializer_list<Neural> layer_list,std::initializer_list<Tensor_Loss_Function> loss_function_list={});




   //Getters
   Tensor Get_Input_Clone() const;

   const Tensor &Get_Input_View() const;

   Tensor& Get_Input_Mutable();

   Tensor Get_Output_Clone() const;

   const Tensor &Get_Output_View() const;

   Tensor& Get_Output_Mutable();

   const Tensor &Get_Downstream_Error() const;

    void Forward_Pass();
    float Train(const Tensor &input, const Tensor &target, float learning_rate, int iterations,int print_every = 10);



   void Modify_Neurones(int layer_number, neurone_intervention_type intervention_type,
                        std::initializer_list<int> neurone_indices,
                        float param1 = -1.0f, float param2 = 1.0f,
                        const Tensor* source = nullptr);

   bool Is_Training_Complete() const;
    bool Is_Forward_Pass_Complete() const;
    bool Is_Backpropagation_Complete() const;
    bool Is_Update_Block_Complete() const;
    bool Is_Local_Cache_Cleared() const;
    bool Is_Block_Cache_Cleared() const;

    void Run_Inference();



private:
    std::vector<Neural> layers_;
    Tensor input_tensor_;
    Tensor output_tensor_;
    Tensor upstream_error_entering_block_;
    Tensor downstream_error_leaving_block_;
    int block_size_;
    std::vector<Tensor_Loss_Function> loss_functions_; // An MLP block can have multiple loss functions

    bool block_constructed;
    bool forward_pass_called_;
    bool back_propagation_done_;
    bool upstream_error_set_;
    bool downstream_error_set_;
    bool update_block_called_;
    bool local_cache_cleared_;
    bool block_cache_cleared_;
    bool training_complete_;
    bool first_inference_called_;


    void Construct_Block_Layers();

    void Perform_Checks() const;

    void BackPropagate(const Tensor &upstream_error_dL_da);

    void Update_Block(float learning_rate);

    void Clear_Local_Cache();

    void Clear_Block_Cache();

    void Reset_Block();

    void Attach_Loss_Heads(const Tensor &prediction);

    float Compute_Loss(const Tensor &prediction, const Tensor &target);


};



#endif //NEURAL_BLOCK_H
