#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = i%10;
    }
    return ten;
}



// __global__ void normalize_(float* input, float* output, int layer_size/*row x col*/,int layer_num/*num_of_matrix*/, float epsilon) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;// compute each layer
//     if (idx < layer_num) {
//         float mean = 0.0f;
//         float variance = 0.0f;
        
//         for (int i = 0; i < layer_size; i++) {
//             mean += input[idx * layer_size + i];
//         }
//         mean /= layer_size;

//         for (int i = 0; i < layer_size; i++) {
//             variance += (input[idx * layer_size + i] - mean) * (input[idx * layer_size + i] - mean);
//         }
        
//         variance /= layer_size;
        
//         for(int i=0; i < layer_size; i++)
//             output[idx * layer_size + i] = (input[idx * layer_size + i] - mean) / sqrtf(variance + epsilon);

//     }
// }


// Tensor* normalize(Tensor* dst, Tensor* src){
//     if(!dst||!src){
//         printf("no Tensor.\n");
//         return NULL;
//     }

//     if(dst->num_dim != src->num_dim){
//         printf("two tensor has different shape.\n");
//         return NULL;
//     }

//     normalize_<<<((dst->sizeTensor/dst->stride[dst->num_dim - 2] + tile_SIZE - 1)/tile_SIZE), tile_SIZE>>>(src->T, dst->T, dst->stride[dst->num_dim - 2], dst->sizeTensor/dst->stride[dst->num_dim - 2], 1e-06);

//     return NULL;
// }

int accuracy_CPU(Tensor* O, Tensor* Y){
    if(!O || !Y){
        printf("no Tensor\n");
        return -1;
    }
    if(O->device_type||Y->device_type){
        printf("CPU ONLY\n");
        return -1;
    }
    if(O->num_dim !=2 || Y->num_dim != 1){
        printf("not an appropriate shape.\n");
        return -1;
    }
    if(O->dim[0] != Y->dim[0]){ //batch 비교
        printf("batch size does not match.\n");
        return -1;
    }
    int acc = 0;
    for(int i=0; i < O->dim[0]; i++){
        int max_inx = 0;
        for(int j=1; j < O->dim[1]; j++){
            if (O->T[i * O->stride[0] + j] > O->T[i * O->stride[0] + max_inx]){
                max_inx = j;
            }

        }
        if(max_inx == Y->T[i]){
            acc++;
        }
    }
    return acc;
}

int main(){
    // Tensor* O = dummyTensor(makeTensor("4, 10", 0));
    // Tensor* Y = makeTensor("4", 0);
    // Y->T[0] = 0;
    // Y->T[1] = 7;
    // Y->T[2] = 9;
    // Y->T[3] = 9;
    // O->T[0] = 100;
    // printTensor(O);
    // printf("%d\n",accuracy_CPU(O, Y));
    Tensor* A = dummyTensor(makeTensor("3 5 2", 0));
    Tensor* subA = makeSubTensor(A, "1 0 0", "5 1");
    printTensor(A);
    printTensor(subA);

}