/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                const int int8_mode,
                                                                gptVariantParams gpt_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params)
{
    mallocWeights();
    setWeightPtr();
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

#ifdef SPARSITY_HAOJUN
template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                // Rewrite construction function
                                                                // Adding 2 new inputs for construction function
                                                                // 1. vector<int> NNZ_List
                                                                // 2. vector<int> SplitK_List
                                                                const std::vector<int> NNZ_List,
                                                                const std::vector<int> NumOffsets_List,
                                                                const std::vector<int> SplitK_List,
                                                                const int int8_mode,
                                                                gptVariantParams gpt_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    NNZ_List_(NNZ_List),
    NumOffsets_List_(NumOffsets_List),
    SplitK_List_(SplitK_List),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params)
{
    mallocWeights();
    setWeightPtr();
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}
#endif

#ifdef SPARSITY_LLM
template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                // Rewrite construction function
                                                                // Adding 5 new inputs for construction function
                                                                const std::vector<int> val_count,
                                                                const std::vector<int> num_gtiles,
                                                                const std::vector<int> num_mtiles,
                                                                const std::vector<int> num_ltiles,
                                                                const std::vector<int> SplitK_List,
                                                                const int int8_mode,
                                                                gptVariantParams gpt_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    val_count_(val_count),
    num_gtiles_(num_gtiles),
    num_mtiles_(num_mtiles),
    num_ltiles_(num_ltiles),
    SplitK_List_(SplitK_List),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params)
{
    mallocWeights();
    setWeightPtr();
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}
#endif

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            if (weights_ptr[i] != nullptr) {
                deviceFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta = nullptr;
        pre_layernorm_weights.gamma = nullptr;
        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;

        after_attention_adapter_weights.intermediate_weight.kernel = nullptr;
        after_attention_adapter_weights.intermediate_weight.bias = nullptr;
        after_attention_adapter_weights.output_weight.kernel = nullptr;
        after_attention_adapter_weights.output_weight.bias = nullptr;

        after_ffn_adapter_weights.intermediate_weight.kernel = nullptr;
        after_ffn_adapter_weights.intermediate_weight.bias = nullptr;
        after_ffn_adapter_weights.output_weight.kernel = nullptr;
        after_ffn_adapter_weights.output_weight.bias = nullptr;

#ifdef SPARSITY_HAOJUN
        for (int i = 0; i < weights_ptr_NZ.size(); i++) {
            if (weights_ptr_NZ[i] != nullptr) {
                deviceFree(weights_ptr_NZ[i]);
            }
        }
        for (int i = 0; i < weights_ptr_Offset.size(); i++) {
            if (weights_ptr_Offset[i] != nullptr) {
                deviceFree(weights_ptr_Offset[i]);
            }
        }
        // NNZ
        self_attention_weights.query_weight.NNZ = -1;
        self_attention_weights.attention_output_weight.NNZ = -1;
        ffn_weights.intermediate_weight.NNZ = -1;
        ffn_weights.output_weight.NNZ = -1;
        // NumOffsets
        self_attention_weights.query_weight.NumOffsets = -1;
        self_attention_weights.attention_output_weight.NumOffsets = -1;
        ffn_weights.intermediate_weight.NumOffsets = -1;
        ffn_weights.output_weight.NumOffsets = -1;
        // SplitK
        self_attention_weights.query_weight.SplitK = -1;
        self_attention_weights.attention_output_weight.SplitK = -1;
        ffn_weights.intermediate_weight.SplitK = -1;
        ffn_weights.output_weight.SplitK = -1;
        // NZWeights
        self_attention_weights.query_weight.NZWeights = nullptr;
        self_attention_weights.attention_output_weight.NZWeights = nullptr;
        ffn_weights.intermediate_weight.NZWeights = nullptr;
        ffn_weights.output_weight.NZWeights = nullptr;
        // TileOffsets
        self_attention_weights.query_weight.TileOffsets = nullptr;
        self_attention_weights.attention_output_weight.TileOffsets = nullptr;
        ffn_weights.intermediate_weight.TileOffsets = nullptr;
        ffn_weights.output_weight.TileOffsets = nullptr;
#endif
#ifdef SPARSITY_LLM
        for (int i = 0; i < weights_ptr_Val.size(); i++) {
            if (weights_ptr_Val[i] != nullptr) {
                deviceFree(weights_ptr_Val[i]);
            }
        }
        for (int i = 0; i < weights_prt_global.size(); i++) {
            if (weights_prt_global[i] != nullptr) {
                deviceFree(weights_prt_global[i]);
            }
        }
        for (int i = 0; i < weights_prt_median.size(); i++) {
            if (weights_prt_median[i] != nullptr) {
                deviceFree(weights_prt_median[i]);
            }
        }
        for (int i = 0; i < weights_prt_bitmap.size(); i++) {
            if (weights_prt_bitmap[i] != nullptr) {
                deviceFree(weights_prt_bitmap[i]);
            }
        }
        for (int i = 0; i < weights_prt_nnz_intile.size(); i++) {
            if (weights_prt_nnz_intile[i] != nullptr) {
                deviceFree(weights_prt_nnz_intile[i]);
            }
        }
        // val_count
        self_attention_weights.query_weight.val_count = -1;
        self_attention_weights.attention_output_weight.val_count = -1;
        ffn_weights.intermediate_weight.val_count = -1;
        ffn_weights.output_weight.val_count = -1;
        // num_gtiles
        self_attention_weights.query_weight.num_gtiles = -1;
        self_attention_weights.attention_output_weight.num_gtiles = -1;
        ffn_weights.intermediate_weight.num_gtiles = -1;
        ffn_weights.output_weight.num_gtiles = -1;
        // num_mtiles
        self_attention_weights.query_weight.num_mtiles = -1;
        self_attention_weights.attention_output_weight.num_mtiles = -1;
        ffn_weights.intermediate_weight.num_mtiles = -1;
        ffn_weights.output_weight.num_mtiles = -1;
        // num_ltiles
        self_attention_weights.query_weight.num_ltiles = -1;
        self_attention_weights.attention_output_weight.num_ltiles = -1;
        ffn_weights.intermediate_weight.num_ltiles = -1;
        ffn_weights.output_weight.num_ltiles = -1;
        // SplitK
        self_attention_weights.query_weight.SplitK = -1;
        self_attention_weights.attention_output_weight.SplitK = -1;
        ffn_weights.intermediate_weight.SplitK = -1;
        ffn_weights.output_weight.SplitK = -1;
        // Compressed_Val_gpu
        self_attention_weights.query_weight.Compressed_Val_gpu = nullptr;
        self_attention_weights.attention_output_weight.Compressed_Val_gpu = nullptr;
        ffn_weights.intermediate_weight.Compressed_Val_gpu = nullptr;
        ffn_weights.output_weight.Compressed_Val_gpu = nullptr;
        // bitmap_TileOffsets_global_gpu
        self_attention_weights.query_weight.bitmap_TileOffsets_global_gpu = nullptr;
        self_attention_weights.attention_output_weight.bitmap_TileOffsets_global_gpu = nullptr;
        ffn_weights.intermediate_weight.bitmap_TileOffsets_global_gpu = nullptr;
        ffn_weights.output_weight.bitmap_TileOffsets_global_gpu = nullptr;
        // bitmap_TileOffsets_median_gpu
        self_attention_weights.query_weight.bitmap_TileOffsets_median_gpu = nullptr;
        self_attention_weights.attention_output_weight.bitmap_TileOffsets_median_gpu = nullptr;
        ffn_weights.intermediate_weight.bitmap_TileOffsets_median_gpu = nullptr;
        ffn_weights.output_weight.bitmap_TileOffsets_median_gpu = nullptr;
        // bitmap_gpu
        self_attention_weights.query_weight.bitmap_gpu = nullptr;
        self_attention_weights.attention_output_weight.bitmap_gpu = nullptr;
        ffn_weights.intermediate_weight.bitmap_gpu = nullptr;
        ffn_weights.output_weight.bitmap_gpu = nullptr;
        // max_nnz_intile
        self_attention_weights.query_weight.max_nnz_intile = nullptr;
        self_attention_weights.attention_output_weight.max_nnz_intile = nullptr;
        ffn_weights.intermediate_weight.max_nnz_intile = nullptr;
        ffn_weights.output_weight.max_nnz_intile = nullptr;
#endif

        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr.size(); i++) {
                if (int8_weights_ptr[i] != nullptr) {
                    deviceFree(int8_weights_ptr[i]);
                }
            }
            for (int i = 0; i < scale_ptr.size(); i++) {
                if (scale_ptr[i] != nullptr) {
                    deviceFree(scale_ptr[i]);
                }
            }
            self_attention_weights.query_weight.int8_kernel = nullptr;
            self_attention_weights.query_weight.scale = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel = nullptr;
            self_attention_weights.attention_output_weight.scale = nullptr;
            ffn_weights.intermediate_weight.int8_kernel = nullptr;
            ffn_weights.intermediate_weight.scale = nullptr;
            ffn_weights.output_weight.int8_kernel = nullptr;
            ffn_weights.output_weight.scale = nullptr;
            after_attention_adapter_weights.intermediate_weight.int8_kernel = nullptr;
            after_attention_adapter_weights.intermediate_weight.scale = nullptr;
            after_attention_adapter_weights.output_weight.int8_kernel = nullptr;
            after_attention_adapter_weights.output_weight.scale = nullptr;
            after_ffn_adapter_weights.intermediate_weight.int8_kernel = nullptr;
            after_ffn_adapter_weights.intermediate_weight.scale = nullptr;
            after_ffn_adapter_weights.output_weight.int8_kernel = nullptr;
            after_ffn_adapter_weights.output_weight.scale = nullptr;
        }

        is_maintain_buffer = false;
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    gpt_variant_params_(other.gpt_variant_params_)
{
#ifdef SPARSITY_HAOJUN
    printf("The copy construction function for ParallelGptDecoderLayerWeight<T> is called.\n ");
    printf("However, this function is not implemented yet!\n");
    exit(-1);
#endif
#ifdef SPARSITY_LLM
    printf("The copy construction function for ParallelGptDecoderLayerWeight<T> is called.\n ");
    printf("However, this function is not implemented yet!\n");
    exit(-1);
#endif
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        cudaD2Dcpy(weights_ptr[12],
                   other.weights_ptr[12],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[14],
                   other.weights_ptr[14],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16],
                   other.weights_ptr[16],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[18],
                   other.weights_ptr[18],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[5], other.scale_ptr[5], hidden_units_);
            cudaD2Dcpy(scale_ptr[6], other.scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[7], other.scale_ptr[7], hidden_units_);
        }
    }

    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>&
ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight& other)
{
#ifdef SPARSITY_HAOJUN
    printf("The copy construction function (operator=) for ParallelGptDecoderLayerWeight<T> is called.\n ");
    printf("However, this function is not implemented yet!\n");
    exit(-1);
#endif
#ifdef SPARSITY_LLM
    printf("The copy construction function (operator=) for ParallelGptDecoderLayerWeight<T> is called.\n ");
    printf("However, this function is not implemented yet!\n");
    exit(-1);
#endif
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    int8_mode_ = other.int8_mode_;
    gpt_variant_params_ = other.gpt_variant_params_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        cudaD2Dcpy(weights_ptr[12],
                   other.weights_ptr[12],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[14],
                   other.weights_ptr[14],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16],
                   other.weights_ptr[16],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[18],
                   other.weights_ptr[18],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[5], other.scale_ptr[5], hidden_units_);
            cudaD2Dcpy(scale_ptr[6], other.scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[7], other.scale_ptr[7], hidden_units_);
        }
    }

    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[0] != 0) {
        loadDataArrayFromBin<unsigned int>(weights_ptr_NZ[0],
                                           NNZ_List_[0],
                                           dir_path + ".attention.query_key_value.weight."
                                               + std::to_string(tensor_para_rank_) + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_ptr_Offset[0],
                                  NumOffsets_List_[0],
                                  dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                      + ".TileOffsets.bin");
    }
    else
    // printf("Printing weights_ptr_Offset[0][i]...\n");
    // int *tmp_intArray_ptr = (int*)malloc(10*sizeof(int));
    // cudaMemcpy(tmp_intArray_ptr, weights_ptr_Offset[0], sizeof(int) *10, cudaMemcpyDeviceToHost);
    // for(int i=0; i<10; i++)
    //    printf("%d ", tmp_intArray_ptr[i] );
    // printf("\n");
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[0] != 0) {
        loadDataArrayFromBin<T>(weights_ptr_Val[0],
                                   val_count_[0],
                                   dir_path + ".attention.query_key_value.weight."
                                               + std::to_string(tensor_para_rank_) + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_prt_global[0],
                                  num_gtiles_[0],
                                  dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                      + ".gtile.bin");  
        loadDataArrayFromBin<int>(weights_prt_median[0],
                                  num_mtiles_[0],
                                  dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                      + ".mtile.bin");
        loadDataArrayFromBin<uint64_t>(weights_prt_bitmap[0],
                                  num_ltiles_[0],
                                  dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                      + ".ltile.bin");
        loadDataArrayFromBin<int>(weights_prt_nnz_intile[0],
                                  1,
                                  dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                      + ".intile.bin");
    }
    else
#endif
        loadWeightFromBin<T>(weights_ptr[2],
                             {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                             dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);

    loadWeightFromBin<T>(weights_ptr[3],
                         {3, hidden_units_ / tensor_para_size_},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[1] != 0) {
        loadDataArrayFromBin<unsigned int>(weights_ptr_NZ[1],
                                           NNZ_List_[1],
                                           dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                               + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_ptr_Offset[1],
                                  NumOffsets_List_[1],
                                  dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                      + ".TileOffsets.bin");
    }
    else
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[1] != 0) {
        loadDataArrayFromBin<T>(weights_ptr_Val[1],
                                   val_count_[1],
                                   dir_path + ".attention.dense.weight."
                                               + std::to_string(tensor_para_rank_) + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_prt_global[1],
                                  num_gtiles_[1],
                                  dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                      + ".gtile.bin");  
        loadDataArrayFromBin<int>(weights_prt_median[1],
                                  num_mtiles_[1],
                                  dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                      + ".mtile.bin");
        loadDataArrayFromBin<uint64_t>(weights_prt_bitmap[1],
                                  num_ltiles_[1],
                                  dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                      + ".ltile.bin");
        loadDataArrayFromBin<int>(weights_prt_nnz_intile[1],
                                  1,
                                  dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                      + ".intile.bin");
    }
    else
#endif
        loadWeightFromBin<T>(weights_ptr[4],
                             {hidden_units_ / tensor_para_size_, hidden_units_},
                             dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

    loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[2] != 0) {
        loadDataArrayFromBin<unsigned int>(weights_ptr_NZ[2],
                                           NNZ_List_[2],
                                           dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                               + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_ptr_Offset[2],
                                  NumOffsets_List_[2],
                                  dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                      + ".TileOffsets.bin");
    }
    else
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[2] != 0) {
        loadDataArrayFromBin<T>(weights_ptr_Val[2],
                                   val_count_[2],
                                   dir_path + ".mlp.dense_h_to_4h.weight."
                                               + std::to_string(tensor_para_rank_) + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_prt_global[2],
                                  num_gtiles_[2],
                                  dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                      + ".gtile.bin");  
        loadDataArrayFromBin<int>(weights_prt_median[2],
                                  num_mtiles_[2],
                                  dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                      + ".mtile.bin");
        loadDataArrayFromBin<uint64_t>(weights_prt_bitmap[2],
                                  num_ltiles_[2],
                                  dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                      + ".ltile.bin");
        loadDataArrayFromBin<int>(weights_prt_nnz_intile[2],
                                  1,
                                  dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                      + ".intile.bin");
    }
    else
#endif
        loadWeightFromBin<T>(weights_ptr[8],
                             {hidden_units_, inter_size_ / tensor_para_size_},
                             dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

    loadWeightFromBin<T>(weights_ptr[9],
                         {inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[3] != 0) {
        loadDataArrayFromBin<unsigned int>(weights_ptr_NZ[3],
                                           NNZ_List_[3],
                                           dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                               + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_ptr_Offset[3],
                                  NumOffsets_List_[3],
                                  dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                      + ".TileOffsets.bin");
    }
    else
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[3] != 0) {
        loadDataArrayFromBin<T>(weights_ptr_Val[3],
                                   val_count_[3],
                                   dir_path + ".mlp.dense_4h_to_h.weight."
                                               + std::to_string(tensor_para_rank_) + ".NZWeights.bin");
        loadDataArrayFromBin<int>(weights_prt_global[3],
                                  num_gtiles_[3],
                                  dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                      + ".gtile.bin");  
        loadDataArrayFromBin<int>(weights_prt_median[3],
                                  num_mtiles_[3],
                                  dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                      + ".mtile.bin");
        loadDataArrayFromBin<uint64_t>(weights_prt_bitmap[3],
                                  num_ltiles_[3],
                                  dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                      + ".ltile.bin");
        loadDataArrayFromBin<int>(weights_prt_nnz_intile[3],
                                  1,
                                  dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                      + ".intile.bin");
    }
    else
#endif
        loadWeightFromBin<T>(weights_ptr[10],
                             {inter_size_ / tensor_para_size_, hidden_units_},
                             dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

    loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

    if (gpt_variant_params_.has_adapters) {
        loadWeightFromBin<T>(weights_ptr[12],
                             {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_attention_adapter.dense_h_to_4h.weight."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[13],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_attention_adapter.dense_h_to_4h.bias."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[14],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                             dir_path + ".after_attention_adapter.dense_4h_to_h.weight."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[15],
                             {hidden_units_},
                             dir_path + ".after_attention_adapter.dense_4h_to_h.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[16],
                             {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_ffn_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[17],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_ffn_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[18],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                             dir_path + ".after_ffn_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[19], {hidden_units_}, dir_path + ".after_ffn_adapter.dense_4h_to_h.bias.bin", model_file_type);
    }

    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta = weights_ptr[0];
    pre_layernorm_weights.gamma = weights_ptr[1];
    self_attention_weights.query_weight.kernel = weights_ptr[2];
    self_attention_weights.query_weight.bias = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias = weights_ptr[5];
    self_attn_layernorm_weights.beta = weights_ptr[6];
    self_attn_layernorm_weights.gamma = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias = weights_ptr[9];
    ffn_weights.output_weight.kernel = weights_ptr[10];
    ffn_weights.output_weight.bias = weights_ptr[11];

    after_attention_adapter_weights.intermediate_weight.kernel = weights_ptr[12];
    after_attention_adapter_weights.intermediate_weight.bias = weights_ptr[13];
    after_attention_adapter_weights.output_weight.kernel = weights_ptr[14];
    after_attention_adapter_weights.output_weight.bias = weights_ptr[15];

    after_ffn_adapter_weights.intermediate_weight.kernel = weights_ptr[16];
    after_ffn_adapter_weights.intermediate_weight.bias = weights_ptr[17];
    after_ffn_adapter_weights.output_weight.kernel = weights_ptr[18];
    after_ffn_adapter_weights.output_weight.bias = weights_ptr[19];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel = int8_weights_ptr[0];
        self_attention_weights.query_weight.scale = scale_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel = int8_weights_ptr[1];
        self_attention_weights.attention_output_weight.scale = scale_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel = int8_weights_ptr[2];
        ffn_weights.intermediate_weight.scale = scale_ptr[2];
        ffn_weights.output_weight.int8_kernel = int8_weights_ptr[3];
        ffn_weights.output_weight.scale = scale_ptr[3];
        after_attention_adapter_weights.intermediate_weight.int8_kernel = int8_weights_ptr[4];
        after_attention_adapter_weights.intermediate_weight.scale = scale_ptr[4];
        after_attention_adapter_weights.output_weight.int8_kernel = int8_weights_ptr[5];
        after_attention_adapter_weights.output_weight.scale = scale_ptr[5];
        after_ffn_adapter_weights.intermediate_weight.int8_kernel = int8_weights_ptr[6];
        after_ffn_adapter_weights.intermediate_weight.scale = scale_ptr[6];
        after_ffn_adapter_weights.output_weight.int8_kernel = int8_weights_ptr[7];
        after_ffn_adapter_weights.output_weight.scale = scale_ptr[7];
    }

#ifdef SPARSITY_HAOJUN
    // setup the pointers and values in struct DenseWeight
    // NNZ
    self_attention_weights.query_weight.NNZ = NNZ_List_[0];
    self_attention_weights.attention_output_weight.NNZ = NNZ_List_[1];
    ffn_weights.intermediate_weight.NNZ = NNZ_List_[2];
    ffn_weights.output_weight.NNZ = NNZ_List_[3];
    // NumOffsets
    self_attention_weights.query_weight.NumOffsets = NumOffsets_List_[0];
    self_attention_weights.attention_output_weight.NumOffsets = NumOffsets_List_[1];
    ffn_weights.intermediate_weight.NumOffsets = NumOffsets_List_[2];
    ffn_weights.output_weight.NumOffsets = NumOffsets_List_[3];
    // SplitK
    self_attention_weights.query_weight.SplitK = SplitK_List_[0];
    self_attention_weights.attention_output_weight.SplitK = SplitK_List_[1];
    ffn_weights.intermediate_weight.SplitK = SplitK_List_[2];
    ffn_weights.output_weight.SplitK = SplitK_List_[3];
    // NZWeights
    self_attention_weights.query_weight.NZWeights = weights_ptr_NZ[0];
    self_attention_weights.attention_output_weight.NZWeights = weights_ptr_NZ[1];
    ffn_weights.intermediate_weight.NZWeights = weights_ptr_NZ[2];
    ffn_weights.output_weight.NZWeights = weights_ptr_NZ[3];
    // TileOffsets
    self_attention_weights.query_weight.TileOffsets = weights_ptr_Offset[0];
    self_attention_weights.attention_output_weight.TileOffsets = weights_ptr_Offset[1];
    ffn_weights.intermediate_weight.TileOffsets = weights_ptr_Offset[2];
    ffn_weights.output_weight.TileOffsets = weights_ptr_Offset[3];
#endif
#ifdef SPARSITY_LLM
    // setup the pointers and values in struct DenseWeight
    // val_count
    self_attention_weights.query_weight.val_count = val_count_[0];
    self_attention_weights.attention_output_weight.val_count = val_count_[1];
    ffn_weights.intermediate_weight.val_count = val_count_[2];
    ffn_weights.output_weight.val_count = val_count_[3];
    // num_gtiles
    self_attention_weights.query_weight.num_gtiles = num_gtiles_[0];
    self_attention_weights.attention_output_weight.num_gtiles = num_gtiles_[1];
    ffn_weights.intermediate_weight.num_gtiles = num_gtiles_[2];
    ffn_weights.output_weight.num_gtiles = num_gtiles_[3];
    // num_mtiles
    self_attention_weights.query_weight.num_mtiles = num_mtiles_[0];
    self_attention_weights.attention_output_weight.num_mtiles = num_mtiles_[1];
    ffn_weights.intermediate_weight.num_mtiles = num_mtiles_[2];
    ffn_weights.output_weight.num_mtiles = num_mtiles_[3];
    // num_ltiles
    self_attention_weights.query_weight.num_ltiles = num_ltiles_[0];
    self_attention_weights.attention_output_weight.num_ltiles = num_ltiles_[1];
    ffn_weights.intermediate_weight.num_ltiles = num_ltiles_[2];
    ffn_weights.output_weight.num_ltiles = num_ltiles_[3];
    // SplitK
    self_attention_weights.query_weight.SplitK = SplitK_List_[0];
    self_attention_weights.attention_output_weight.SplitK = SplitK_List_[1];
    ffn_weights.intermediate_weight.SplitK = SplitK_List_[2];
    ffn_weights.output_weight.SplitK = SplitK_List_[3];
    // Compressed_Val_gpu
    self_attention_weights.query_weight.Compressed_Val_gpu = weights_ptr_Val[0];
    self_attention_weights.attention_output_weight.Compressed_Val_gpu = weights_ptr_Val[1];
    ffn_weights.intermediate_weight.Compressed_Val_gpu = weights_ptr_Val[2];
    ffn_weights.output_weight.Compressed_Val_gpu = weights_ptr_Val[3];
    // bitmap_TileOffsets_global_gpu
    self_attention_weights.query_weight.bitmap_TileOffsets_global_gpu = weights_prt_global[0];
    self_attention_weights.attention_output_weight.bitmap_TileOffsets_global_gpu = weights_prt_global[1];
    ffn_weights.intermediate_weight.bitmap_TileOffsets_global_gpu = weights_prt_global[2];
    ffn_weights.output_weight.bitmap_TileOffsets_global_gpu = weights_prt_global[3];
    // bitmap_TileOffsets_median_gpu
    self_attention_weights.query_weight.bitmap_TileOffsets_median_gpu = weights_prt_median[0];
    self_attention_weights.attention_output_weight.bitmap_TileOffsets_median_gpu = weights_prt_median[1];
    ffn_weights.intermediate_weight.bitmap_TileOffsets_median_gpu = weights_prt_median[2];
    ffn_weights.output_weight.bitmap_TileOffsets_median_gpu = weights_prt_median[3];
    // bitmap_gpu
    self_attention_weights.query_weight.bitmap_gpu = weights_prt_bitmap[0];
    self_attention_weights.attention_output_weight.bitmap_gpu = weights_prt_bitmap[1];
    ffn_weights.intermediate_weight.bitmap_gpu = weights_prt_bitmap[2];
    ffn_weights.output_weight.bitmap_gpu = weights_prt_bitmap[3];
    // max_nnz_intile
    self_attention_weights.query_weight.max_nnz_intile = weights_prt_nnz_intile[0];
    self_attention_weights.attention_output_weight.max_nnz_intile = weights_prt_nnz_intile[1];
    ffn_weights.intermediate_weight.max_nnz_intile = weights_prt_nnz_intile[2];
    ffn_weights.output_weight.max_nnz_intile = weights_prt_nnz_intile[3];
#endif
    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[0] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_HAOJUN is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[0] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_LLM is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[1] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_HAOJUN is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[1] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_LLM is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
        deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_);
    deviceMalloc(&weights_ptr[7], hidden_units_);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[2] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_HAOJUN is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[2] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_LLM is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
        deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
#ifdef SPARSITY_HAOJUN
    if (SplitK_List_[3] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_HAOJUN is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
#ifdef SPARSITY_LLM
    if (SplitK_List_[3] == 0)  // Do not malloc for 4 dense matrix each decoder layer if SPARSITY_LLM is detected,
                               // unless SplitK==0 which means it is kept as dense during pruning
#endif
        deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);

#ifdef SPARSITY_HAOJUN
    // Malloc memory for 4 sparse matrix, 8 memory space reserved for both NNZWeights and TileOffsets
    // printf("Definition of SPARSITY_HAOJUN: %d\n", SPARSITY_HAOJUN);
    FT_CHECK_WITH_INFO(NNZ_List_.size() == 4 && NumOffsets_List_.size() == 4,
                       "ERROR: NNZ_List_ and NumOffsets_List_ should have 4 elements!\n");
    for (int i = 0; i < 4; i++) {
        if (SplitK_List_[i] != 0) {
            check_cuda_error(cudaMalloc((void**)(&weights_ptr_NZ[i]), sizeof(unsigned int) * NNZ_List_[i]));
            check_cuda_error(cudaMalloc((void**)(&weights_ptr_Offset[i]), sizeof(unsigned int) * NumOffsets_List_[i]));
        }
    }
#endif
#ifdef SPARSITY_LLM
    // Malloc memory for 4 sparse matrix, 20 memory space reserved for both NNZWeights and TileOffsets
    // printf("Definition of SPARSITY_LLM: %d\n", SPARSITY_LLM);
    FT_CHECK_WITH_INFO(val_count_.size() == 4 && num_gtiles_.size() == 4 && num_mtiles_.size() == 4 && num_ltiles_.size() == 4, 
                       "ERROR: val_count_, num_gtiles_, num_mtiles_ and num_ltiles_ should have 4 elements!\n");
    for (int i = 0; i < 4; i++) {
        if (SplitK_List_[i] != 0) {
            check_cuda_error(cudaMalloc((void**)(&weights_ptr_Val[i]), sizeof(T) * val_count_[i]));
            check_cuda_error(cudaMalloc((void**)(&weights_prt_global[i]), sizeof(int) * num_gtiles_[i]));
            check_cuda_error(cudaMalloc((void**)(&weights_prt_median[i]), sizeof(int) * num_mtiles_[i]));
            check_cuda_error(cudaMalloc((void**)(&weights_prt_bitmap[i]), sizeof(uint64_t) * num_ltiles_[i]));
            check_cuda_error(cudaMalloc((void**)(&weights_prt_nnz_intile[i]), sizeof(int) * 1));
        }
    }
#endif
    if (gpt_variant_params_.has_adapters) {
        deviceMalloc(&weights_ptr[12], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[14], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        deviceMalloc(&weights_ptr[16], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[18], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        deviceMalloc(&scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[1], hidden_units_);
        deviceMalloc(&scale_ptr[2], inter_size_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[3], hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            deviceMalloc(&int8_weights_ptr[4],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[5],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&int8_weights_ptr[6],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[7],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&scale_ptr[5], hidden_units_);
            deviceMalloc(&scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&scale_ptr[7], hidden_units_);
        }
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelGptDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_ = 4 * hidden_units_;

    const size_t num_sparse_weights = 8;
    size_t shapes[num_sparse_weights][2] = {
        {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
        {hidden_units_ / tensor_para_size_, hidden_units_},
        {hidden_units_, inter_size_ / tensor_para_size_},
        {inter_size_ / tensor_para_size_, hidden_units_},
        {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
        {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
        {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
        {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_}};

    const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
                                                  self_attention_weights.attention_output_weight.kernel,
                                                  ffn_weights.intermediate_weight.kernel,
                                                  ffn_weights.output_weight.kernel,
                                                  after_attention_adapter_weights.intermediate_weight.kernel,
                                                  after_attention_adapter_weights.output_weight.kernel,
                                                  after_ffn_adapter_weights.intermediate_weight.kernel,
                                                  after_ffn_adapter_weights.output_weight.kernel};

    size_t real_num_sparse_weights = gpt_variant_params_.has_adapters ? num_sparse_weights : (num_sparse_weights - 4);
    for (size_t i = 0; i < real_num_sparse_weights; ++i) {
        int m = shapes[i][1];
        int k = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[3];
    after_attention_adapter_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    after_attention_adapter_weights.output_weight.sp_kernel = sp_weights_ptr[5];
    after_ffn_adapter_weights.intermediate_weight.sp_kernel = sp_weights_ptr[6];
    after_ffn_adapter_weights.output_weight.sp_kernel = sp_weights_ptr[7];
    is_maintain_sp_buffer = true;
}
#endif

template<typename T>
void ParallelGptDecoderLayerWeight<T>::transposeCalibrateQuantizeWeight()
{
    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[0], weights_ptr[2], hidden_units_, 3 * hidden_units_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[0],
                                               scale_ptr[0],
                                               weights_ptr[2],
                                               hidden_units_,
                                               3 * hidden_units_ / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[1], scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[2], scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[3], scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[4],
                                       weights_ptr[12],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[4],
                                               scale_ptr[4],
                                               weights_ptr[12],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[5],
                                       weights_ptr[14],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[5],
                                               scale_ptr[5],
                                               weights_ptr[14],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[6],
                                       weights_ptr[16],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[6],
                                               scale_ptr[6],
                                               weights_ptr[16],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[7],
                                       weights_ptr[18],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[7],
                                               scale_ptr[7],
                                               weights_ptr[18],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelGptDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
