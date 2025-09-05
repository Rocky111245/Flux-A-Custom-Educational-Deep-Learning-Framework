// =================================
// Created by Rocky170 on 9/05/2025.
// =================================

#include "Tensor_Loss_Functions.h"



constexpr float EPSILON = 1e-7f;


// ===========================
// Tensor_Loss_Function class
// ===========================


float Tensor_Loss_Function::Compute(const Tensor& predicted, const Tensor& target, const float huber_delta) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Predicted/target shapes must match in Tensor_Loss_Function::Compute .");
    }

    const int R = predicted.rows();
    const int C = predicted.columns();
    const int D = predicted.depth();

    downstream_error_ = Tensor(R, C, D);

    switch (type_) {
        case tensor_loss_function::NONE: {
            loss_value_ = 0.0f;
            downstream_error_.Fill(0.0f);
            break;
        }

        case tensor_loss_function::MSE_LOSS: {
            loss_value_ = Tensor_Mean_Squared_Error(predicted, target);
            const float invN2 = 2.0f / static_cast<float>(R * C * D);
            for (int d = 0; d < D; ++d)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        float diff = predicted(r, c, d) - target(r, c, d);
                        downstream_error_(r, c, d) = invN2 * diff;
                    }
            break;
        }
        case tensor_loss_function::MAE_LOSS: {
            loss_value_ = Tensor_Mean_Absolute_Error(predicted, target);
            const float invN = 1.0f / static_cast<float>(R * C * D);
            for (int d = 0; d < D; ++d)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        float diff = predicted(r, c, d) - target(r, c, d);
                        float g = (diff > 0.0f) ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                        downstream_error_(r, c, d) = invN * g;
                    }
            break;
        }
        case tensor_loss_function::HUBER_LOSS: {
            if (huber_delta <= 0.0f)
                throw std::invalid_argument("huber_delta must be > 0");
            loss_value_ = Tensor_Huber_Loss(predicted, target, huber_delta);
            const float invN = 1.0f / static_cast<float>(R * C * D);
            for (int d = 0; d < D; ++d)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        float e = predicted(r, c, d) - target(r, c, d);
                        float ae = std::fabs(e);
                        float g = (ae <= huber_delta) ? e : (huber_delta * (e > 0.0f ? 1.0f : -1.0f));
                        downstream_error_(r, c, d) = invN * g;
                    }
            break;
        }
        case tensor_loss_function::CROSS_ENTROPY_LOSS: {
            loss_value_ = Tensor_Binary_Cross_Entropy(predicted, target);
            const float invN = 1.0f / static_cast<float>(R * C * D);
            for (int d = 0; d < D; ++d)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        float y = target(r, c, d);
                        float p = std::clamp(predicted(r, c, d), EPSILON, 1.0f - EPSILON);
                        float grad = ((1.0f - y) / (1.0f - p)) - (y / p);
                        downstream_error_(r, c, d) = invN * grad;
                    }
            break;
        }
        case tensor_loss_function::CATEGORICAL_CROSS_ENTROPY_LOSS: {
            loss_value_ = Tensor_Categorical_Cross_Entropy(predicted, target);
            const float invN = 1.0f / static_cast<float>(R * D);
            for (int d = 0; d < D; ++d)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c) {
                        float y = target(r, c, d);
                        if (y > 0.0f) {
                            float p = std::clamp(predicted(r, c, d), EPSILON, 1.0f - EPSILON);
                            downstream_error_(r, c, d) = -invN * (y / p);
                        } else {
                            downstream_error_(r, c, d) = 0.0f;
                        }
                    }
            break;
        }
        default:
            throw std::runtime_error("Unknown tensor_loss_function enum variant.");
    }

    return loss_value_;
}




// ===========================
// Scalar helper functions
// ===========================

float Tensor_Mean_Squared_Error(const Tensor& predicted, const Tensor& target) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Tensor shapes do not match for MSE.");
    }

    float sum = 0.0f;
    int total = predicted.rows() * predicted.columns() * predicted.depth();

    for (int d = 0; d < predicted.depth(); ++d)
        for (int r = 0; r < predicted.rows(); ++r)
            for (int c = 0; c < predicted.columns(); ++c) {
                float diff = predicted(r, c, d) - target(r, c, d);
                sum += diff * diff;
            }

    return sum / total;
}

float Tensor_Mean_Absolute_Error(const Tensor& predicted, const Tensor& target) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Tensor shapes do not match for MAE.");
    }

    float sum = 0.0f;
    int total = predicted.rows() * predicted.columns() * predicted.depth();

    for (int d = 0; d < predicted.depth(); ++d)
        for (int r = 0; r < predicted.rows(); ++r)
            for (int c = 0; c < predicted.columns(); ++c)
                sum += std::abs(predicted(r, c, d) - target(r, c, d));

    return sum / total;
}

float Tensor_Huber_Loss(const Tensor& predicted, const Tensor& target, const float delta) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Tensor shapes do not match for Huber loss.");
    }

    float loss = 0.0f;
    int total = predicted.rows() * predicted.columns() * predicted.depth();

    for (int d = 0; d < predicted.depth(); ++d)
        for (int r = 0; r < predicted.rows(); ++r)
            for (int c = 0; c < predicted.columns(); ++c) {
                float error = predicted(r, c, d) - target(r, c, d);
                float abs_err = std::fabs(error);
                if (abs_err <= delta) {
                    loss += 0.5f * error * error;
                } else {
                    loss += delta * (abs_err - 0.5f * delta);
                }
            }

    return loss / total;
}

float Tensor_Binary_Cross_Entropy(const Tensor& predicted, const Tensor& target) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Tensor shapes do not match for BCE.");
    }

    float loss = 0.0f;
    int total = predicted.rows() * predicted.columns() * predicted.depth();

    for (int d = 0; d < predicted.depth(); ++d)
        for (int r = 0; r < predicted.rows(); ++r)
            for (int c = 0; c < predicted.columns(); ++c) {
                float y = target(r, c, d);
                float y_hat = std::clamp(predicted(r, c, d), EPSILON, 1.0f - EPSILON);
                loss += y * std::log(y_hat) + (1.0f - y) * std::log(1.0f - y_hat);
            }

    return -loss / total;
}

float Tensor_Categorical_Cross_Entropy(const Tensor& predicted, const Tensor& target) {
    if (predicted.rows() != target.rows() ||
        predicted.columns() != target.columns() ||
        predicted.depth() != target.depth()) {
        throw std::invalid_argument("Tensor shapes do not match for categorical cross entropy.");
    }

    float loss = 0.0f;
    int batch = predicted.depth();
    int seq_len = predicted.rows();
    int num_classes = predicted.columns();

    for (int d = 0; d < batch; ++d)
        for (int r = 0; r < seq_len; ++r)
            for (int c = 0; c < num_classes; ++c) {
                float y = target(r, c, d);
                float y_hat = std::clamp(predicted(r, c, d), EPSILON, 1.0f - EPSILON);
                if (y > 0.0f)
                    loss += y * std::log(y_hat);
            }

    return -loss / (seq_len * batch);
}
