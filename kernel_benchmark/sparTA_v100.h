/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
 #include "./sputnik_utils.h"
 #include "sputnik/sputnik.h"
 // 移除cuSPARSELt头文件，因为V100不兼容
 // #include <cusparseLt.h>
 
 #define SPARTA_M 2
 #define SPARTA_N 4
 #define WARM_UP_ITERATION 10
 #define BENCHMARK_ITERATION 100
 
 void transform(half* A_h, half* A1_h, half* A2_h, int length)
 {
     // split the matrix A into A1 and A2
     // A1 is for the saprse tensor core, A2 is for the finegrained sparse kernel
     memset(A1_h, 0, sizeof(half) * length);
     memset(A2_h, 0, sizeof(half) * length);
     assert(length % SPARTA_N == 0);
     int nnz = 0;
     for (int i = 0; i < length / SPARTA_N; i++) {
         int start = i * SPARTA_N;
         int end   = start + SPARTA_N;
         nnz       = 0;
         for (int j = start; j < end; j++) {
             if (fabs(__half2float(A_h[j])) > 0.0000001) {
                 if (nnz < SPARTA_M) {
                     A1_h[j] = A_h[j];
                 }
                 else {
                     A2_h[j] = A_h[j];
                 }
                 nnz++;
             }
         }
     }
     int NNZ_A  = 0;
     int NNZ_A1 = 0;
     int NNZ_A2 = 0;
     for (int i = 0; i < length; i++)
         if (fabs(__half2float(A_h[i])) > 0.0000001)
             NNZ_A++;
     for (int i = 0; i < length; i++)
         if (fabs(__half2float(A1_h[i])) > 0.0000001)
             NNZ_A1++;
     for (int i = 0; i < length; i++)
         if (fabs(__half2float(A2_h[i])) > 0.0000001)
             NNZ_A2++;
     printf("NZ_A: %2.3f \t NZ_A1: %2.3f \t NZ_A2: %2.3f\n",
            float(NNZ_A) / length,
            float(NNZ_A1) / length,
            float(NNZ_A2) / length);
 }
 
 void check_A1_h(half* A, int length)
 {
     assert(length % 4 == 0);
     for (int i = 0; i < length / 4; i++) {
         int nnz = 0;
         for (int j = 0; j < 4; j++) {
             if (__half2float(A[i * 4 + j]) != 0.0f)
                 nnz++;
         }
         if (nnz > 2) {
             printf("Failed to meet 2:4 requirements!\n");
             exit(-1);
         }
     }
     printf("Succeeded in meeting 2:4 requirements!\n");
 }
 
 // V100兼容的sparTA版本 - 只使用Sputnik部分，跳过cuSPARSELt
 int sparTA_v100(half* A_h, half* B_h, half* C_h, int m, int n, int k, float* milliseconds)
 {
     half *A1_h, *A2_h;
     A1_h = (half*)malloc(sizeof(half) * m * k);
     if (A1_h == NULL) {
         printf("Error in sparTA_v100.h: line %d malloc failed\n", __LINE__);
         exit(-1);
     }
     A2_h = (half*)malloc(sizeof(half) * m * k);
     if (A2_h == NULL) {
         printf("Error in sparTA_v100.h: line %d malloc failed\n", __LINE__);
         exit(-1);
     }
     
     transform(A_h, A1_h, A2_h, m * k);
     
     half *A1_d, *B_d, *C_d;
     CHECK_CUDA(cudaMalloc((void**)&A1_d, m * k * sizeof(half)))
     CHECK_CUDA(cudaMalloc((void**)&B_d, k * n * sizeof(half)))
     CHECK_CUDA(cudaMalloc((void**)&C_d, m * n * sizeof(half)))
     CHECK_CUDA(cudaMemcpy(A1_d, A1_h, m * k * sizeof(half), cudaMemcpyHostToDevice))
     CHECK_CUDA(cudaMemcpy(B_d, B_h, k * n * sizeof(half), cudaMemcpyHostToDevice))
     CHECK_CUDA(cudaMemset(C_d, 0, m * n * sizeof(half)))
 
     // 只使用Sputnik部分，跳过cuSPARSELt
     float* A_float_h = (float*)malloc(sizeof(float) * m * k);
     for (int i = 0; i < m * k; i++)
         A_float_h[i] = __half2float(A2_h[i]);
     
     sputnik_utils::SparseMatrix            sparse_matrix(m, k, A_float_h, sputnik_utils::IDENTITY, 4);
     sputnik_utils::CudaSparseMatrix<half2> sparse_matrix_gpu(sparse_matrix);
     
     // 执行矩阵乘法
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     
     // 预热
     for (int i = 0; i < WARM_UP_ITERATION; i++) {
         CUDA_CALL(sputnik::CudaSpmm(m,
                                     k,
                                     n,
                                     sparse_matrix_gpu.NumElementsWithPadding(),
                                     sparse_matrix_gpu.RowIndices(),
                                     sparse_matrix_gpu.Values(),
                                     sparse_matrix_gpu.RowOffsets(),
                                     sparse_matrix_gpu.ColumnIndices(),
                                     reinterpret_cast<half2*>(B_d),
                                     reinterpret_cast<half2*>(C_d),
                                     0));
     }
     
     // 基准测试
     cudaEventRecord(start);
     for (int i = 0; i < BENCHMARK_ITERATION; i++) {
         CUDA_CALL(sputnik::CudaSpmm(m,
                                     k,
                                     n,
                                     sparse_matrix_gpu.NumElementsWithPadding(),
                                     sparse_matrix_gpu.RowIndices(),
                                     sparse_matrix_gpu.Values(),
                                     sparse_matrix_gpu.RowOffsets(),
                                     sparse_matrix_gpu.ColumnIndices(),
                                     reinterpret_cast<half2*>(B_d),
                                     reinterpret_cast<half2*>(C_d),
                                     0));
     }
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(milliseconds, start, stop);
     *milliseconds = *milliseconds / BENCHMARK_ITERATION;
     
     // 清理
     cudaFree(A1_d);
     cudaFree(B_d);
     cudaMemcpy(C_h, C_d, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
     cudaFree(C_d);
     
     free(A1_h);
     free(A2_h);
     free(A_float_h);
     
     return 0;
 }