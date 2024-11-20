// //nvcc cublas.cu -lcublas
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <stdio.h>

// #define M 2048  // Number of rows in matrix A and C
// #define N 2048  // Number of columns in matrix B and C
// #define K 2048  // Number of columns in matrix A and rows in matrix B

// // Error checking macro
// #define CUDA_CHECK(status) \
//     if (status != cudaSuccess) { \
//         printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
//         return EXIT_FAILURE; \
//     }

// #define CUBLAS_CHECK(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         printf("cuBLAS Error\n"); \
//         return EXIT_FAILURE; \
//     }

// int main() {
//     // Initialize cuBLAS handle
//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

//     // Allocate host memory
//     float *h_A = (float *)malloc(M * K * sizeof(float));
//     float *h_B = (float *)malloc(K * N * sizeof(float));
//     float *h_C = (float *)malloc(M * N * sizeof(float));

//     // Initialize matrices with random values
//     for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
//     for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;
//     for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

//     // Allocate device memory
//     float *d_A, *d_B, *d_C;
//     CUDA_CHECK(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
//     CUDA_CHECK(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
//     CUDA_CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

//     // Copy data from host to device
//     CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

//     // Set alpha and beta values
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Perform matrix multiplication: C = alpha * A * B + beta * C
//     // A: M x K, B: K x N, C: M x N
//     CUBLAS_CHECK(cublasSgemm(
//         handle,
//         CUBLAS_OP_N,  // No transpose for A
//         CUBLAS_OP_N,  // No transpose for B
//         N,            // Number of columns of matrix C and B
//         M,            // Number of rows of matrix C and A
//         K,            // Number of columns of matrix A and rows of matrix B
//         &alpha,       // Scalar alpha
//         d_B, N,       // Matrix B and its leading dimension
//         d_A, K,       // Matrix A and its leading dimension
//         &beta,        // Scalar beta
//         d_C, N        // Matrix C and its leading dimension
//     ));

//     // Copy the result back to host
//     CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

//     // Print a small part of the result matrix
//     printf("C[0,0] = %f\n", h_C[0]);
//     printf("C[1,1] = %f\n", h_C[N + 1]);
//     printf("C[2,2] = %f\n", h_C[2 * N + 2]);

//     // Clean up resources
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     free(h_A);
//     free(h_B);
//     free(h_C);
//     cublasDestroy(handle);

//     return 0;
// }

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 5 // Number of rows in A and C
#define N 4 // Number of columns in B and C
#define K 3 // Number of columns in A and rows in B
#define BATCH_SIZE 4

int main() {
    // Leading dimensions
    int lda = K;
    int ldb = M;
    int ldc = K;

    // Strides between matrices in the batch
    long long int strideA = M * K;
    long long int strideB = 0;        // Since B is broadcasted
    long long int strideC = M * N;

    // Allocate host memory for matrices A, B, and C
    float h_A[BATCH_SIZE * strideA];
    float h_B[K * N];
    float h_C[BATCH_SIZE * strideC];

    // Initialize host matrices A and B
    for(int i=0; i < BATCH_SIZE * strideA; i++){
        h_A[i] = i;
    }

    for(int i=0; i < K * N; i++){
        h_B[i] = i;
    }



    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, BATCH_SIZE * strideA * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, BATCH_SIZE * strideC * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_A, h_A, BATCH_SIZE * strideA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set up scalar values
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform batched matrix multiplication with broadcasting of B
    cublasStatus_t stat = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, // Transpose option for A
        CUBLAS_OP_T, // Transpose option for B
        M,           // Number of rows in A and C
        N,           // Number of columns in B and C
        K,           // Number of columns in A and rows in B
        &alpha,      // Scalar alpha
        d_B,         // Device pointer to matrix B
        ldb,         // Leading dimension of B
        strideB,     // Stride between B_i and B_(i+1) (0 for broadcasting)
        d_A,         // Device pointer to matrix A
        lda,         // Leading dimension of A
        strideA,     // Stride between A_i and A_(i+1)
        &beta,       // Scalar beta
        d_C,         // Device pointer to matrix C
        ldc,         // Leading dimension of C
        strideC,     // Stride between C_i and C_(i+1)
        BATCH_SIZE   // Number of matrices in the batch
    );

    // Check for errors
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemmStridedBatched failed\n");
        return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, BATCH_SIZE * strideC * sizeof(float), cudaMemcpyDeviceToHost);

    // Destroy cuBLAS handle and free device memory
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the results
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        printf("Result of batch %d:\n", batch);
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                int idx = batch * strideC + col * ldc + row;
                printf("%6.2f ", h_C[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
