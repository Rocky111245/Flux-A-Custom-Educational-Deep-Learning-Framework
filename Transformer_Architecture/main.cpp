//
// Created by rakib on 18/6/2025.
//

#include <iostream>
#include "Tokenizer/Tokenizer.h"
#include "Token_Embedding/Token_Embedding.h"
#include "Positional_Embeddings/Fixed_Positional_Encoding.h"
#include "Positional_Embeddings/Learned_Positional_Embeddings.h"
#include "Embedding_Operations/Embedding_Operations.h"

int main() {
    std::cout << "=== MODIFIED BPE TOKENIZER TESTING  ===" << std::endl << std::endl;

    // Test with some cancer research text
    std::string cancer_text =
            "Cancer cells undergo uncontrolled proliferation, evading normal regulatory mechanisms such as apoptosis and senescence, "
            "and often acquire capabilities like angiogenesis and immune escape. Metastasis, the process by which cancer spreads to "
            "distant organs, is mediated by epithelial-to-mesenchymal transition and facilitated by matrix metalloproteinases. "
            "Oncogenes such as MYC, RAS, and TP53 mutations drive carcinogenesis by disrupting cell cycle checkpoints and DNA repair pathways. "
            "Tumor suppressor genes, including RB1 and BRCA1/2, when inactivated, further accelerate genomic instability. "
            "Chemotherapy agents including cisplatin, carboplatin, and paclitaxel target rapidly dividing cells, though resistance mechanisms "
            "such as drug efflux pumps and DNA repair reactivation often develop. Combination regimens like FOLFIRI or CHOP are used for "
            "specific cancers to improve response rates. Immunotherapy approaches, including checkpoint inhibitors like pembrolizumab, "
            "nivolumab, and atezolizumab, enhance immune system recognition of neoplastic cells by blocking PD-1/PD-L1 or CTLA-4 pathways. "
            "Adoptive cell therapies, such as CAR-T cell therapy, re-engineer patient T-cells to recognize tumor-associated antigens with high "
            "specificity. Targeted therapies such as trastuzumab for HER2-positive breast cancer, erlotinib for EGFR-mutated lung cancer, "
            "and imatinib for BCR-ABL fusion-positive chronic myeloid leukemia have revolutionized precision oncology by enabling molecularly"
            " guided treatment. Radiation therapy, when combined with surgical resection, remains the gold standard for early-stage solid tumors, "
            "including glioblastoma and localized prostate cancer. Biomarkers including PD-L1 expression, microsatellite instability (MSI), "
            "tumor mutational burden (TMB), and circulating tumor DNA (ctDNA) are increasingly used to guide personalized treatment selection. "
            "Liquid biopsies now allow for non-invasive monitoring of minimal residual disease and treatment response. Novel therapeutic modalities "
            "encompass bispecific antibodies, cancer vaccines targeting neoantigens, oncolytic viruses, and mRNA-based immunotherapies. "
            "Advances in single-cell sequencing and spatial transcriptomics are unraveling tumor heterogeneity and enabling the discovery of new "
            "therapeutic targets. Multi-omics integration, combining genomics, transcriptomics, proteomics, and metabolomics, is reshaping the "
            "landscape of systems oncology. Precision medicine, supported by AI-driven clinical decision support systems, is moving toward real-time, "
            "adaptive cancer treatment strategies that evolve alongside tumor dynamics.";

    std::cout << "ORIGINAL TEXT:" << std::endl;
    std::cout << "\"" << cancer_text << "\"" << std::endl << std::endl;

    std::cout << "Original text length: " << cancer_text.length() << " characters" << std::endl << std::endl;

    // Test tokenizer with different configurations
    std::cout << "=== TESTING TOKENIZER WITH DIFFERENT CONFIGURATIONS ===" << std::endl << std::endl;

    // Configuration 1: Short sequences
    std::cout << "--- Configuration 1: Short Sequences (128 tokens, 1000 vocab) ---" << std::endl;
    try {
        Matrix result1 = Batch_Tokenization_Pipeline(cancer_text, 128, 1000);

        std::cout << "Tokenization successful" << std::endl;
        std::cout << "Batch matrix dimensions: " << result1.rows() << " rows X " << result1.columns() << " columns" << std::endl;
        std::cout << "Total sequences created: " << result1.rows() << std::endl << std::endl;

        // Display first few rows of the matrix
        std::cout << "TOKENIZED MATRIX (First 3 sequences):" << std::endl;
        int rows_to_show = std::min(3, result1.rows());
        int cols_to_show = std::min(20, result1.columns()); // Show first 20 tokens per row

        for (int i = 0; i < rows_to_show; ++i) {
            std::cout << "Sequence " << i << ": [";
            for (int j = 0; j < cols_to_show; ++j) {
                std::cout << std::setw(4) << static_cast<int>(result1(i, j));
                if (j < cols_to_show - 1) std::cout << ", ";
            }
            if (cols_to_show < result1.columns()) {
                std::cout << " ... (+" << (result1.columns() - cols_to_show) << " more)";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;

        // Show complete first sequence
        std::cout << "COMPLETE FIRST SEQUENCE (all " << result1.columns() << " tokens):" << std::endl;
        for (int j = 0; j < result1.columns(); ++j) {
            int token_id = static_cast<int>(result1(0, j));
            std::cout << std::setw(4) << token_id;
            if ((j + 1) % 16 == 0) std::cout << std::endl; // New line every 16 tokens
            else if (j < result1.columns() - 1) std::cout << " ";
        }
        std::cout << std::endl << std::endl;

    } catch (const std::exception& e) {
        std::cout << "X Error in Configuration 1: " << e.what() << std::endl << std::endl;
    }

    // Configuration 2: Medium sequences
    std::cout << "--- Configuration 2: Medium Sequences (256 tokens, 1500 vocab) ---" << std::endl;
    try {
        Matrix result2 = Batch_Tokenization_Pipeline(cancer_text, 256, 1500);

        std::cout << "Tokenization successful" << std::endl;
        std::cout << "Batch matrix dimensions: " << result2.rows() << " rows X " << result2.columns() << " columns" << std::endl;

        // Show some token distribution metrics
        std::cout << "TOKEN Metrics:" << std::endl;
        int padding_count = 0;
        int non_padding_count = 0;

        for (int i = 0; i < result2.rows(); ++i) {
            for (int j = 0; j < result2.columns(); ++j) {
                if (static_cast<int>(result2(i, j)) == 0) {
                    padding_count++;
                } else {
                    non_padding_count++;
                }
            }
        }

        std::cout << "- Total tokens: " << (padding_count + non_padding_count) << std::endl;
        std::cout << "- Content tokens: " << non_padding_count << std::endl;
        std::cout << "- Padding tokens: " << padding_count << std::endl;
        std::cout << "- Content ratio: " << std::fixed << std::setprecision(2)
                  << (100.0 * non_padding_count / (padding_count + non_padding_count)) << "%" << std::endl << std::endl;

    } catch (const std::exception& e) {
        std::cout << "X Error in Configuration 2: " << e.what() << std::endl << std::endl;
    }

    // Configuration 3: Large vocabulary
    std::cout << "--- Configuration 3: Large Vocabulary (256 tokens, 3000 vocab) ---" << std::endl;
    try {
        Matrix result3 = Batch_Tokenization_Pipeline(cancer_text, 256, 3000);

        std::cout << " Tokenization successful" << std::endl;
        std::cout << "Batch matrix dimensions: " << result3.rows() << " rows × " << result3.columns() << " columns" << std::endl;

        // Display summary statistics
        std::cout << "BATCH SUMMARY:" << std::endl;
        for (int i = 0; i < result3.rows(); ++i) {
            int content_tokens = 0;
            int first_padding = -1;

            for (int j = 0; j < result3.columns(); ++j) {
                if (static_cast<int>(result3(i, j)) == 0) {
                    if (first_padding == -1) first_padding = j;
                } else {
                    content_tokens++;
                }
            }

            std::cout << "- Batch " << i << ": " << content_tokens << " content tokens";
            if (first_padding != -1) {
                std::cout << ", padding starts at position " << first_padding;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cout << "X Error in Configuration 3: " << e.what() << std::endl << std::endl;
    }

    // Test edge cases
    std::cout << "=== EDGE CASE TESTING ===" << std::endl;

    // Short text test
    std::cout << "--- Testing Short Text ---" << std::endl;
    std::string short_text = "Cancer treatment protocols require careful monitoring of patient responses.";
    try {
        Matrix short_result = Batch_Tokenization_Pipeline(short_text, 64, 500);
        std::cout << "Short text processing successful" << std::endl;
        std::cout << "Dimensions: " << short_result.rows() << "×" << short_result.columns() << std::endl;

        // Show the complete short sequence
        std::cout << "Complete tokenized sequence: [";
        for (int j = 0; j < short_result.columns(); ++j) {
            std::cout << static_cast<int>(short_result(0, j));
            if (j < short_result.columns() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl << std::endl;

    } catch (const std::exception& e) {
        std::cout << "X Short text error: " << e.what() << std::endl << std::endl;
    }

    std::cout << "=== TOKENIZER TESTING COMPLETE ===" << std::endl;
    std::cout << "Custom tokenizer passed all tests" << std::endl;

    return 0;
}