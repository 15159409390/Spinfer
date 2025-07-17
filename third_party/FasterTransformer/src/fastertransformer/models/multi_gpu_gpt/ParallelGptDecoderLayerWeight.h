/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <string>

#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

struct gptVariantParams {
    // GPT default params
    float layernorm_eps = 1e-6f;
    LayerNormType layernorm_type = LayerNormType::pre_layernorm;
    ActivationType activation_type = ActivationType::Gelu;
    bool has_post_decoder_layernorm = true;
    // detoxification adapters. refer to
    bool has_adapters = false;
    size_t adapter_inter_size = 0;
};

template<typename T>
struct ParallelGptDecoderLayerWeight {
public:
    ParallelGptDecoderLayerWeight() = default;
    ParallelGptDecoderLayerWeight(const int int8_mode);
    ParallelGptDecoderLayerWeight(const int hidden_units,
                                  const int inter_size,
                                  const int tensor_para_size,
                                  const int tensor_para_rank,
                                  const int int8_mode = 0,
                                  gptVariantParams gpt_variant_params = {});

#ifdef SPARSITY_HAOJUN
    ParallelGptDecoderLayerWeight(const int hidden_units,
                                  const int inter_size,
                                  const int tensor_para_size,
                                  const int tensor_para_rank,
                                  const std::vector<int> NNZ_List,
                                  const std::vector<int> NumOffsets_List,
                                  const std::vector<int> SplitK_List,
                                  const int int8_mode = 0,
                                  gptVariantParams gpt_variant_params = {});
#endif

#ifdef SPARSITY_LLM
    ParallelGptDecoderLayerWeight(const int hidden_units,
                                  const int inter_size,
                                  const int tensor_para_size,
                                  const int tensor_para_rank,
                                  const std::vector<int> val_count,
                                  const std::vector<int> num_gtiles,
                                  const std::vector<int> num_mtiles,
                                  const std::vector<int> num_ltiles,
                                  const std::vector<int> SplitK_List,
                                  const int int8_mode = 0,
                                  gptVariantParams gpt_variant_params = {});
#endif
    ~ParallelGptDecoderLayerWeight();
    ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other);
    ParallelGptDecoderLayerWeight& operator=(const ParallelGptDecoderLayerWeight& other);
    void loadModel(std::string dir_path, FtCudaDataType model_file_type);
#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif
    void transposeCalibrateQuantizeWeight();

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    FfnWeight<T> ffn_weights;
    FfnWeight<T> after_attention_adapter_weights;
    FfnWeight<T> after_ffn_adapter_weights;

private:
    void setWeightPtr();
    void mallocWeights();

protected:
    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_ = 1;
    size_t tensor_para_rank_ = 0;
    bool is_maintain_buffer = false;
    int int8_mode_ = 0;

    // gpt varians params. e.g. detoxification adapters
    gptVariantParams gpt_variant_params_;

    std::vector<T*> weights_ptr = std::vector<T*>(20, nullptr);

#ifdef SPARSITY_HAOJUN
    const std::vector<int> NNZ_List_;
    const std::vector<int> NumOffsets_List_;
    const std::vector<int> SplitK_List_;
    // 2 ptr for each sparse matrix, 4 sparse matrix for each decoder layer
    // first 4 PTRs for NZWeight, next 4 PTRs for TileOffsets
    std::vector<unsigned int*> weights_ptr_NZ = std::vector<unsigned int*>(4, nullptr);
    std::vector<int*> weights_ptr_Offset = std::vector<int*>(4, nullptr);
#endif

#ifdef SPARSITY_LLM
    const std::vector<int> val_count_;
    const std::vector<int> num_gtiles_;
    const std::vector<int> num_mtiles_;
    const std::vector<int> num_ltiles_;
    const std::vector<int> SplitK_List_;
    // 5 ptr for each sparse matrix, 4 sparse matrix for each decoder layer
    std::vector<T*> weights_ptr_Val = std::vector<T*>(4, nullptr);
    std::vector<int*> weights_prt_global = std::vector<int*>(4, nullptr);
    std::vector<int*> weights_prt_median = std::vector<int*>(4, nullptr);
    std::vector<uint64_t*> weights_prt_bitmap = std::vector<uint64_t*>(4, nullptr);
    std::vector<int*> weights_prt_nnz_intile = std::vector<int*>(4, nullptr);
#endif

    std::vector<int8_t*> int8_weights_ptr = std::vector<int8_t*>(8, nullptr);

    std::vector<float*> scale_ptr = std::vector<float*>(8, nullptr);
    cudaStream_t stream_ = 0;

#ifdef SPARSITY_ENABLED
    std::vector<T*> sp_weights_ptr = std::vector<T*>(8, nullptr);
    bool is_maintain_sp_buffer = false;
#endif
};

}  // namespace fastertransformer
