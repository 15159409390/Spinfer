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
#include "stdlib.h"
#include <cstdint>      // For uint64_t
namespace fastertransformer {

template<typename T>
struct DenseWeight {
    const T* kernel = nullptr;
    const T* bias = nullptr;
    const T* sp_kernel = nullptr;
    // for int8 kernel
    const int8_t* int8_kernel = nullptr;
    const float* scale = nullptr;
    //
#ifdef SPARSITY_HAOJUN
    int SplitK = -1;                          // SplitK
    int NNZ = -1;                             // Number of Non-Zeros in this spase matrix
    int NumOffsets = -1;                      // Number of TileOffsets after padding (+2)
    const unsigned int* NZWeights = nullptr;  // Non-Zero Weights, NZWeights[NNZ]
    const int* TileOffsets = nullptr;         // TileOffsets[ (M/TILE_M)*(K/TILE_K)+2 ]
#endif
#ifdef SPARSITY_LLM
    int SplitK = -1;                          // SplitK
    int val_count = -1;                       // Length of Compressed_Val_gpu (Number of Non-Zeros in this spase matrix)
    int num_gtiles = -1;                      // Length of bitmap_TileOffsets_global_gpu
    int num_mtiles = -1;                      // Length of bitmap_TileOffsets_median_gpu
    int num_ltiles = -1;                      // Length of bitmap_gpu
    const T* Compressed_Val_gpu = nullptr;  
    const int* bitmap_TileOffsets_global_gpu = nullptr;         
    const int* bitmap_TileOffsets_median_gpu = nullptr;
    const uint64_t* bitmap_gpu = nullptr;
    const int* max_nnz_intile = nullptr;      // The address of max_nnz_intile, just one number per matrix
#endif
};

}  // namespace fastertransformer