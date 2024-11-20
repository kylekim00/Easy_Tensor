#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

void row_major_gemm(cublasHandle_t handle, float* A, float* B, float* C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // 리딩 다이멘션 설정
    int lda = M;  // A의 리딩 다이멘션 (row-major에서 행 개수)
    int ldb = K;  // B의 리딩 다이멘션 (row-major에서 행 개수)
    int ldc = M;  // C의 리딩 다이멘션 (row-major에서 행 개수)

    // cuBLAS 호출: row-major 데이터를 column-major로 해석하기 위해 트랜스포즈 옵션 사용
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed with error code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int M = 4;
    const int K = 3;
    const int N = 5;

    float h_A[M * K] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    float h_B[K * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };

    float h_C[M * N] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    row_major_gemm(handle, d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C (row-major):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
