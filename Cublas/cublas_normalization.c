#include <cuda_runtime.h>
#include <stdio.h>
#include<stdlib.h>

// Include the normalize_tensor_cublas function here or in a header file

int main() {
    // Tensor size
    int N = 1024;

    // Allocate host memory
    float* h_tensor = (float*)malloc(N * sizeof(float));

    // Initialize the tensor with some values
    for (int i = 0; i < N; ++i) {
        h_tensor[i] = (float)(i + 1);  // Example values
    }

    // Allocate device memoryz
    float* d_tensor;
    CUDA_CHECK(cudaMalloc((void**)&d_tensor, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_tensor, h_tensor, N * sizeof(float), cudaMemcpyHostToDevice));

    // Normalize the tensor
    normalize_tensor_cublas(d_tensor, N);

    // Copy the normalized tensor back to host
    CUDA_CHECK(cudaMemcpy(h_tensor, d_tensor, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: Verify the result by computing the norm on the host
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += h_tensor[i] * h_tensor[i];
    }
    float host_norm = sqrtf(sum);
    printf("Host computed norm after normalization: %f\n", host_norm);  // Should be close to 1.0

    // Free memory
    free(h_tensor);
    CUDA_CHECK(cudaFree(d_tensor));

    return 0;
}
