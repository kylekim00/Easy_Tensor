#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include <math.h>

#include "easy_tensor.h"
//나중에는 나눠서 각각의 gpu 안에 넣어야하기 때문에 생각을 해보면 인덱스 값에 따라 값을 copy 해주는 것도 있으면 좋을 것 같다.
//기존 텐서와 다른점. 

//1. dim과 stride가 다를 때를 고려한다. 아울러 총 크기를 그냥 저장해버린다. 이는 makelightcopysubTensor() 함수를 만들기 위함이다. 마치 커서의 드래그와 같은 역할을 해줄 것이다.(해결)
//2. num_dim이 1, 2 일때도 작동이 되도록한다.
//3. device 위에 올라가 있을 경우 dim과 stride도 같이 device에 올려준다. [5, 1, 3, 3, 4] X [4, 1, 4, 2] 와 같은 복잡한 텐서도 행렬곱이 가능하게 하기 위함이다. 



Tensor *mallocTensor(int *dim, int num_dim, int device_type){
    if(!dim){                       //if There is no dim inside
        return NULL;
    }
    if(num_dim < 0){                //if there is not an appropriate num_dim
        return NULL;
    }
    int device;
    cudaGetDeviceCount(&device);
    if(device_type > device){       //count device and check the boundary
        printf("mallocTensor : DEVICE NUM %d NOT AVAILABLE\n", device_type);
        return NULL;
    }
    

    int sizeTensor;                 //check the size of whole Tensor

    Tensor* ten = (Tensor*)malloc(sizeof(Tensor));      //give tensor a space for host
    ten->dim = (int*)malloc(2 * num_dim * sizeof(int)); //give dim and stride a spcae for host
    ten->stride = ten->dim+num_dim;                    //this approach might be effective when sending to GPU later.

    ten->num_dim = num_dim;
    ten->device_type = device_type;

    sizeTensor = 1;
    for(int i= num_dim - 1; i >= 0; i--){
        ten->dim[i] = dim[i];
        ten->stride[i] = sizeTensor;
        sizeTensor *= dim[i];
    }

    ten->sizeTensor = sizeTensor;

    if(!device_type){
        ten->T = (float*)malloc(sizeTensor * sizeof(float));
        ten->d_dim_stride = NULL;
    }else{
        cudaSetDevice(device_type-1);
        cudaMalloc(&ten->T, sizeTensor * sizeof(float));
        cudaMalloc(&ten->d_dim_stride, 2 * num_dim * sizeof(int));
        cudaMemcpy(ten->d_dim_stride, ten->dim, 2 * num_dim * sizeof(int), cudaMemcpyHostToDevice);
    }
    ten->isSub = 0;
    ten->dT = NULL;
    return ten;
}

Tensor* makeTensor(const char dim[], int device_type) {  // Use `const char[]`
    int dim_[MAX_NUM_DIM];  // Array to store dimensions
    int num_dim = 0;        // Counter for the number of dimensions

    const char *ptr = dim;  // Pointer to traverse the string (now `const`)

    // Skip leading spaces
    while (*ptr == ' ') {
        ptr++;
    }

    // Parse the dimension string
    while (*ptr != '\0' && num_dim < MAX_NUM_DIM) {  // Ensure we don't exceed max dims
        // Skip spaces and commas between numbers
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }

        if (*ptr == '\0') {
            break;  // End of string
        }

        int value = 0;
        int sign = 1;

        // Optional: Handle negative numbers
        if (*ptr == '-') {
            sign = -1;
            ptr++;
        }

        // Convert digit characters to integer
        while (*ptr >= '0' && *ptr <= '9') {
            value = value * 10 + (*ptr - '0');
            ptr++;
        }

        // Store the parsed number in the dimensions array
        dim_[num_dim++] = sign * value;

        // Skip any spaces or commas after the number
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }
    }

    // If no dimensions were parsed, return NULL or handle the error appropriately
    if (num_dim == 0) {
        printf("makeTensor : No valid dimensions were parsed.\n");
        return NULL;
    }
    // Call makeTensor with the parsed dimensions
    return mallocTensor(dim_, num_dim, device_type);
}


Tensor* makeTensorbyShape(Tensor* src, int device_type){
    if(!src){
        printf("makeTensorbyShape : SouRCe is vacant.\n");
        return NULL;
    }
    // if(src->isSub){
    //     printf("Source is SubTensor.\n");
    //     return NULL;
    // }
    return mallocTensor(src->dim, src->num_dim, device_type);
}


Tensor* mallocSubTensor(Tensor* src, int* start_point, int* dim, int num_dim){
    if(src->isSub){
        printf("mallocSubTensor : Can't light copy subTensor\n");
        return NULL;
    }
    if(src->num_dim < num_dim){
        printf("mallocSubTensor : SouRCe num_dim not that big\n");
        return NULL;
    }

    int cont = src->num_dim - num_dim;
    float* sp = src->T;


    for(int i=0; i < src->num_dim; i++){                            //This is where you set the starting point
        if(src->dim[i] <= start_point[i]){
            printf("mallocSubTensor : starting point invalid.\n");
            return NULL;
        }
        sp += start_point[i] * src->stride[i];
    }

    for(int i=0; i < num_dim; i++){                                 //This is where tou check the size of the dim 
        if(src->dim[i + cont] < start_point[i+cont] + dim[i]){
            printf("mallocSubTensor : SouRCe not that big.\n");
            return NULL;
        }
    }

    Tensor* subTensor = (Tensor*)malloc(sizeof(Tensor));            //Tensor malloc

    subTensor->isSub = 1;
    subTensor->device_type = src->device_type;                      //device_type same as src
    subTensor->num_dim = num_dim;                                   //num_dim

    subTensor->dim = (int*)malloc(2 * sizeof(int)* num_dim);        //dim stride malloc
    subTensor->stride = subTensor->dim + num_dim;

    subTensor->sizeTensor = 1;
    for(int i=0; i < num_dim; i++){                             
        subTensor->dim[i] = dim[i];
        subTensor-> stride[i] = src->stride[i+cont];            //copy Stride
        subTensor->sizeTensor *= dim[i];
    }

    subTensor->T = sp;

    if(src->device_type){
        cudaSetDevice(src->device_type-1);
        cudaMalloc(&subTensor->d_dim_stride, 2 * num_dim * sizeof(int));
        cudaMemcpy(subTensor->d_dim_stride, subTensor->dim, 2 * num_dim * sizeof(int), cudaMemcpyHostToDevice);
    }else{
        subTensor->d_dim_stride = NULL;
    }

    return subTensor;
}

Tensor* makeSubTensor(Tensor* src, const char start_point[], const char dim[]){
    if(!src){
        printf("makeSubTensor : no SouRCe Tensor.\n");
        return NULL;
    }
    int start_point_[MAX_NUM_DIM];  // Array to store dimensions
    int num_sp = 0;        // Counter for the number of dimensions

    const char *ptr = start_point;  // Pointer to traverse the string (now `const`)

    // Skip leading spaces
    while (*ptr == ' ') {
        ptr++;
    }

    // Parse the dimension string
    while (*ptr != '\0') {  // Ensure we don't exceed max dims
        // Skip spaces and commas between numbers
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }

        if (*ptr == '\0') {
            break;  // End of string
        }

        int value = 0;
        int sign = 1;

        // Optional: Handle negative numbers
        if (*ptr == '-') {
            sign = -1;
            ptr++;
        }

        // Convert digit characters to integer
        while (*ptr >= '0' && *ptr <= '9') {
            value = value * 10 + (*ptr - '0');
            ptr++;
        }

        // Store the parsed number in the dimensions array
        start_point_[num_sp++] = sign * value;

        // Skip any spaces or commas after the number
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }
    }

    // If no dimensions were parsed, return NULL or handle the error appropriately
    if (num_sp != src->num_dim) {
        printf("makeSubTensor : not an appropriate dimension number for starting point");
        return NULL;
    }

    ///////////////////////////////////////////////////////////////////////
    int dim_[MAX_NUM_DIM];  // Array to store dimensions
    int num_dim = 0;        // Counter for the number of dimensions

    ptr = dim;  // Pointer to traverse the string (now `const`)

    // Skip leading spaces
    while (*ptr == ' ') {
        ptr++;
    }

    // Parse the dimension string
    while (*ptr != '\0' && num_dim < MAX_NUM_DIM) {  // Ensure we don't exceed max dims
        // Skip spaces and commas between numbers
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }

        if (*ptr == '\0') {
            break;  // End of string
        }

        int value = 0;
        int sign = 1;

        // Optional: Handle negative numbers
        if (*ptr == '-') {
            sign = -1;
            ptr++;
        }

        // Convert digit characters to integer
        while (*ptr >= '0' && *ptr <= '9') {
            value = value * 10 + (*ptr - '0');
            ptr++;
        }

        // Store the parsed number in the dimensions array
        dim_[num_dim++] = sign * value;

        // Skip any spaces or commas after the number
        while (*ptr == ' ' || *ptr == ',') {
            ptr++;
        }
    }

    // If no dimensions were parsed, return NULL or handle the error appropriately
    if (num_dim == 0) {
        printf("makeSubTensor : No valid dimensions were parsed.\n");
        return NULL;
    }

    return mallocSubTensor(src, start_point_, dim_, num_dim);
}

//================================================FREEEEEEEEEEEE===============================================================

void freeSubTensor(Tensor* subTen){
    if(!subTen->isSub){
        printf("freeSubTensor : This is Not a SubTensor.\n");
        return;
    }
    if(subTen==NULL){
        printf("freeSubTensor : NO TENSOR IN POINTER.\n");
        return;
    }else{
        if(subTen->device_type){
            cudaSetDevice(subTen->device_type - 1);
            cudaError_t err = cudaFree(subTen->d_dim_stride);
            if (err != cudaSuccess) {
                fprintf(stderr, "freeSubTensor CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }
        free(subTen->dim);         //didn't malloc ten->stride from the first place :P
        free(subTen);
    }
}
void freeTensor(Tensor *ten){
    if(ten==NULL){
        printf("freeTensor : NO TENSOR IN POINTER.\n");
        return;
    }
    if(ten->isSub){
        freeSubTensor(ten);
        return;
    }else{
        if(ten->device_type){
            cudaSetDevice(ten->device_type - 1);
            cudaError_t err = cudaFree(ten->T);
            if (err != cudaSuccess) {
                fprintf(stderr, "freeTensor CUDA Error: %s\n", cudaGetErrorString(err));
            }
            err = cudaFree(ten->d_dim_stride);
            if (err != cudaSuccess) {
                fprintf(stderr, "freeTensor CUDA Error: %s\n", cudaGetErrorString(err));
            }
            if(ten->dT){
                cudaError_t err = cudaFree(ten->dT);
                // printf(" 이거냐 \n");
                if (err != cudaSuccess) {
                    fprintf(stderr, "freeTensor CUDA Error: %s\n", cudaGetErrorString(err));
                }
            }
        }else{
            free(ten->T);
        }
    }
    free(ten->dim);         //didn't malloc ten->stride from the first place :P
    free(ten);
}

//=================================print================================

Tensor* infoTensor(Tensor *ten){
    if(!ten){
        printf("infoTensor : Tensor is NULL.\n");
        return NULL;
    }
    printf("\n=========Tensor===========\n");
    if(ten->isSub){
        printf("===SUBTENSOR===\n");
    }
    printf("DIMENSION : [");
    for(int i=0; i < ten->num_dim-1; i++){
        printf("%d ", ten->dim[i]);
    }
    printf("%d]\n", ten->dim[ten->num_dim - 1]);
    printf("STRIDE    : [");
    for(int i=0; i < ten->num_dim-1; i++){
        printf("%d ", ten->stride[i]);
    }
    printf("%d]\n", ten->stride[ten->num_dim - 1]);
    printf("DEVICE TYPE : ");
    if(ten->device_type){
        printf("GPU %d", ten->device_type);
    }else{
        printf("CPU");
    }
    printf("\n==========================\n");
    return ten;
}

Tensor* printTensor(Tensor *ten){
    if(!ten){
        printf("printTensor : No tensor.\n");
        return NULL;
    }
    if(ten->device_type){
        printf("printTensor : GPU mem can not be printed\n");
        return ten;
    }
    infoTensor(ten);
    //==============if ten->num_dim < 3========================
    if(ten->num_dim == 1){
        printf("[ %d ]\n", ten->dim[0]);
        for(int i=0; i < ten->dim[0]; i+=ten->stride[0]){
            printf("%.03f\t", ten->T[i]);
        }
        printf("\n");
        return ten;
    }
    if(ten->num_dim == 2){
        printf("[ %d %d ]\n", ten->dim[0], ten->dim[1]);
        for(int i=0; i < ten->dim[0]*ten->stride[0]; i+=ten->stride[0]){
            for(int j=0; j < ten->dim[1]*ten->stride[1]; j+=ten->stride[1]){
                printf("%.03f\t", ten->T[i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }
        printf("\n");
        return ten;
    }
    //=================else====================================
    printf("=\n");

    int* tmp_Inx = (int*)malloc(sizeof(int) * (ten->num_dim - 2));
    for(int i=0; i < ten->num_dim - 2; i++){
        tmp_Inx[i] = 0;
    }
    int inx;
    while(tmp_Inx[0] < ten->dim[0]){
        inx = 0;
        printf("[ ");
        for(int i=0; i < ten->num_dim-2;i++){
            printf("%d ", tmp_Inx[i]);
            inx += tmp_Inx[i] * ten->stride[i];
        }
        printf("- - ]\n");

        for(int i=0; i < ten->dim[ten->num_dim-2]*ten->stride[ten->num_dim-2]; i+=ten->stride[ten->num_dim-2]){
            for(int j=0; j < ten->dim[ten->num_dim-1]*ten->stride[ten->num_dim-1]; j+=ten->stride[ten->num_dim-1]){
                printf("%.03f\t", ten->T[inx + i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }

        tmp_Inx[ten->num_dim - 3]++;
        for(int i = ten->num_dim - 3; i > 0; i--){
            if(tmp_Inx[i] >= ten->dim[i]){
                tmp_Inx[i-1]++;
                tmp_Inx[i] = 0;
            }
        }
    }
    printf("=\n");
    free(tmp_Inx);
    return ten;
}

Tensor* copyTensor(Tensor *dst, Tensor *src){
    if(!dst || !src){
        printf("copyTensor : No dst or src\n");
        return NULL;
    }
    if(dst->isSub||src->isSub){
        printf("copyTensor : dst or src is subTensor.");
        return NULL;
    }
    if(dst->num_dim != src->num_dim){
        printf("copyTensor : shape of dst and src doesn't match.\n");
        return NULL;
    }
    for(int i=0; i < dst->num_dim; i++){
        if(dst->dim[i] != src->dim[i]){
            printf("copyTensor : shape of dst and src doesn't match.\n");
            return NULL;
        }
    }
    if(!dst->device_type && !src->device_type){ //CPU to CPU
        for(int i=0; i < dst->dim[0]*dst->stride[0]; i++)
            dst->T[i] = src->T[i];

    }

    else if(dst->device_type && src->device_type){
        // cudaSetDevice(dst->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else if(dst->device_type){
        cudaSetDevice(dst->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyHostToDevice);
    }else{
        cudaSetDevice(src->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return dst;
}


__global__ void reshape_(float* dst, float* src, int* dst_dim_stride, int* src_dim_stride, int* reshape, int num_dim, int sizeTensor){
    src_dim_stride += num_dim;
    dst_dim_stride += num_dim;

    int new_inx = blockDim.x * blockIdx.x + threadIdx.x;
    int new_tmp = new_inx;
    int inx = 0;
    for(int i=0; i < num_dim; i++){
        inx += new_tmp / dst_dim_stride[i] * src_dim_stride[reshape[i]];
        new_tmp %= dst_dim_stride[i];
    }
    if(new_inx < sizeTensor)                            //This is for dst because tile size could go over the sizeTensor.
        dst[new_inx] = src[inx];


}


Tensor* copyReshapeTensor(Tensor* dst, Tensor* src, int* reshape){
    if(!dst || !src){
        printf("copyReshapeTensor : no Tensor\n");
        return NULL;
    }
    if(dst->isSub){
        printf("copyReshapeTensor : dst can't be subMatrix.\n");
        return NULL;
    }
    if(src->device_type != dst->device_type){
        printf("copyReshapeTensor : DEVICE NOT MATCH.\n");
        return NULL;
    }
    if(src->num_dim != dst->num_dim){
        printf("copyReshapeTensor : DEVICE NUM_DIM DOES NOT MATCH.\n");
        return NULL;
    }

    if(src->sizeTensor != dst->sizeTensor){
        printf("copyReshapeTensor : DEVICE NUM OF ELEMENT DOES NOT MATCH.\n");
        return NULL;
    }

    if(src->device_type){//GPU
        cudaSetDevice(src->device_type - 1);
        int* d_tmp_reshape;
        cudaMalloc(&d_tmp_reshape, sizeof(int) * dst->num_dim);
        cudaMemcpy(d_tmp_reshape, reshape, sizeof(int) * dst->num_dim, cudaMemcpyHostToDevice);
        
        int s_tile_SIZE = tile_SIZE * tile_SIZE * 2;//no special drawbacks in parallel sequence, so just put the values into the tiles linearly.
        
        reshape_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, src->T, dst->d_dim_stride, src->d_dim_stride, d_tmp_reshape, src->num_dim, dst->sizeTensor);
        cudaFree(d_tmp_reshape);

    }else{//CPU
        int newInx, inx;
        for(int new_inx =0; new_inx < dst->sizeTensor; new_inx++){
            newInx = new_inx;
            inx = 0;
            for(int i=0; i < src->num_dim; i++){
                inx += newInx / dst->stride[i] * src->stride[reshape[i]];
                newInx = newInx % dst->stride[i];
            }
            dst->T[new_inx] = src->T[inx];
        }
    }

    return dst;
}


//빠르긴 한데....
__global__ void transposeCoalesced(float *odata, const float *idata, int dst_row, int dst_col, int src_col_stride, int dst_mat_stride, int src_mat_stride){
    __shared__ float tile[tile_SIZE][tile_SIZE];
    int x = blockIdx.y * tile_SIZE + threadIdx.x;  // 이건 dst를 기준으로 만든것. 일단 dim은 똑같기 때문에 조심해서 해보자. 
    int y = blockIdx.x * tile_SIZE + threadIdx.y;
    int z = blockIdx.z;
    if(x < dst_row)
        for (int j = 0; j < tile_SIZE && (y+j) < dst_col; j++)
            tile[threadIdx.x][threadIdx.y + j] = idata[z * src_mat_stride + (y+j)*src_col_stride + x];//src_row는 stride[num_dim - 2] 그리고 어차피 필요한건 다 저장이 되기 때문에 굳이 쓰레긱 값에 집중하지 말자.
    __syncthreads();
    x = blockIdx.x * tile_SIZE + threadIdx.x;
    y = blockIdx.y * tile_SIZE + threadIdx.y;

    // if(x ==8 && y==0 && z==0){
    //     for(int i=0; i < tile_SIZE; i++){
    //         for(int j=0; j < tile_SIZE; j++){
    //             printf("%.02f ", tile[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }
    
    if(x < dst_col){
        for (int j = 0; j < tile_SIZE && y+j < dst_row; j++)
            odata[z * dst_mat_stride + (y+j) * dst_col + x] = tile[threadIdx.y+j][threadIdx.x];
    }
}

//////////////////////////////////////////////////////////////////////////////////

Tensor* copyTransposeTensor(Tensor* dst, Tensor* src){
    if(dst->device_type != src->device_type){
        printf("copyTransposeTensor : device_type does not match.\n");
        return NULL;
    }
    if(dst->num_dim <2){
        printf("copyTransposeTensor : dimension lesser than 2 dimension need no Transpose.\n");
        return NULL;
    }
    if(dst->num_dim != src->num_dim){
        printf("copyTransposeTensor : num_dim does not match.(%d)(%d)\n", dst->num_dim, src->num_dim);
        return NULL;
    }
    
    for(int i=0; i < dst->num_dim - 2; i++){
        if(dst->dim[i] != src->dim[i]){
            printf("copyTransposeTensor : dimension %d does not match.\n", i);
            return NULL;
        }
    }
    if(dst->dim[dst->num_dim - 1] != src->dim[src->num_dim-2] || dst->dim[dst->num_dim - 2] != src->dim[src->num_dim-1]){
        printf("copyTransposeTensor : row and col does not match.\n");
        return NULL;
    }
    

    if(dst->isSub){
        printf("copyTransposeTensor : dst sub Matrix not allowed\n");
        return NULL;
    }
    
    if(dst->device_type){
        int dim_mat_dst;
        int dim_mat_src;
        int z;

        if(dst->num_dim == 2){
            dim_mat_dst = dst->sizeTensor;
            dim_mat_src = src->sizeTensor;
            z = 1;
        }else{
            dim_mat_dst = dst->stride[dst->num_dim - 3];
            dim_mat_src = src->stride[dst->num_dim - 3];
            z = dst->sizeTensor / dst->stride[dst->num_dim - 3];
        }
        cudaSetDevice(dst->device_type-1);
        dim3 dimGrid((dst->dim[dst->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dst->dim[dst->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, z); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
        dim3 dimBlock(tile_SIZE,1);

        transposeCoalesced<<<dimGrid, dimBlock>>>(dst->T, src->T, dst->dim[dst->num_dim - 2], dst->dim[dst->num_dim - 1], src->stride[src->num_dim - 2], dim_mat_dst, dim_mat_src);
    }else{
        if(dst->num_dim == 1){
            for(int i=0; i < dst->sizeTensor; i++){
                dst->dim[i] = src->dim[i];
            }
        }
        else if(dst->num_dim == 2){
            for(int i=0; i < dst->dim[0]; i++){
                for(int j=0; j < dst->dim[1]; j++){
                    dst->T[i*dst->dim[1] + j] = src->T[j*src->stride[0] + i];
                }
            }
        }
        else{
            
            for(int k=0; k < dst->sizeTensor / dst->stride[dst->num_dim - 3]; k++){
                for(int i=0; i < dst->dim[dst->num_dim - 2]; i++){
                    for(int j=0; j < dst->dim[dst->num_dim - 1]; j++){
                        dst->T[k * dst->stride[dst->num_dim - 3] + i * dst->stride[dst->num_dim - 2] + j] = src->T[k * src->stride[src->num_dim - 3] + j * src->stride[src->num_dim - 2] + i];
                    }
                }
            }
        }
    }
    return dst;

}

//이거 num_dim 3이상이어야 한다.
//This has to have Z dim value of C->sizeTensor / C->stride[C->num_dim - 2]
__global__ void compTiledMM_Abig(float*A, float *B, float *C, float *bias, int *dimA, int *dimB, int *dimC, int num_dim, int little_num_dim, char bias_row){
    int matIdx_A = 0, matIdx_B = 0;
    int matIdx_C = blockDim.z * blockIdx.z; //We multiply matrixDim because We have to calculate the IdxA, IdxB.
    int cont = num_dim - little_num_dim;

 int matIdx_tmp = matIdx_C * dimC[num_dim + num_dim - 3];

for(int i=0; i < cont; i++){             //To matrix(이거 그냥 가져다 쓰면 된다. 뭐 곱할 필요 없음.)
    if(dimA[i] != 1){
        matIdx_A += (matIdx_tmp/dimC[num_dim + i]) * dimA[num_dim + i];// dim inx 값을 도출하여 각각의 stride를 곱해준다. 
    }
    matIdx_tmp %= dimC[num_dim + i];
}

for(int i=cont; i < num_dim - 2; i++){      //To matrix
    if(dimA[i] != 1){
        matIdx_A += (matIdx_tmp/dimC[num_dim + i]) * dimA[num_dim + i];// dim inx 값을 도출하여 각각의 stride를 곱해준다. 
    }
    if(dimB[i-cont] != 1){
        matIdx_B += (matIdx_tmp/dimC[num_dim + i]) * dimB[little_num_dim + i-cont];
    }
    matIdx_tmp %= dimC[num_dim + i];
}
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;
    
    //output 매트릭스 stride는 그냥 dim에 맞게 해라. 그게 맞다,,,, 귀찮게 하지 말고.....

    for (int i = 0; i < (dimA[num_dim - 1] + tile_SIZE - 1) / tile_SIZE; i++) {
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdx_A = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
        if (row < dimA[num_dim - 2] && (i * tile_SIZE + threadIdx.x) < dimA[num_dim - 1])
            s_a[threadIdx.y][threadIdx.x] = A[matIdx_A + row * dimA[2*num_dim - 2] + (i * tile_SIZE + threadIdx.x)];
        else 
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        

        if (col < dimB[little_num_dim-1] && (i * tile_SIZE + threadIdx.y) < dimA[num_dim-1])
            s_b[threadIdx.y][threadIdx.x] = B[matIdx_B  + (i * tile_SIZE + threadIdx.y) * dimB[2*little_num_dim - 2] + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    // printf("[%d %d] [%d %d]\n", row, col, dimC[num_dim - 2], dimC[num_dim-1]);
    // if (row < dimC[num_dim - 2] && col < dimC[num_dim - 1]) {
    //     int bias_row = 0;
    //     if (bias)
    //         tmp = tmp + bias[(bias_row?row:col)];
    //     // printf("[%d %d %d] [%d %d] : %f\n", matIdx_C,row, col, dimC[num_dim - 2], dimC[num_dim-1], tmp);
    //     C[matIdx_C * dimC[2*num_dim - 3] + row * dimC[2*num_dim - 2] + col] = tmp;
    // }
    if (row < dimC[num_dim - 2] && col < dimC[num_dim - 1]) {
        if (bias)
            tmp = tmp + bias[(bias_row?row:col)];
        C[matIdx_C * dimC[2*num_dim - 3] + row * dimC[2*num_dim - 2] + col] = tmp;
    }
}

__global__ void compTiledMM_Bbig(float*A, float *B, float *C, float *bias, int *dimA, int *dimB, int *dimC, int little_num_dim, int num_dim, char bias_row){
    int matIdx_A = 0, matIdx_B = 0;
    int matIdx_C = blockDim.z * blockIdx.z; //We multiply matrixDim because We have to calculate the IdxA, IdxB.
    int cont = num_dim - little_num_dim;

 int matIdx_tmp = matIdx_C * dimC[num_dim + num_dim - 3];

for(int i=0; i < cont; i++){             //To matrix(이거 그냥 가져다 쓰면 된다. 뭐 곱할 필요 없음.)
    if(dimB[i] != 1){
        matIdx_B += (matIdx_tmp/dimC[num_dim + i]) * dimB[num_dim + i];// dim inx 값을 도출하여 각각의 stride를 곱해준다. 
    }
    matIdx_tmp %= dimC[num_dim + i];
}

for(int i=cont; i < num_dim - 2; i++){      //To matrix
    if(dimA[i-cont] != 1){
        matIdx_A += (matIdx_tmp/dimC[num_dim + i]) * dimA[little_num_dim + i-cont];// dim inx 값을 도출하여 각각의 stride를 곱해준다. 
    }
    if(dimB[i] != 1){
        matIdx_B += (matIdx_tmp/dimC[num_dim + i]) * dimB[num_dim + i];
    }
    matIdx_tmp %= dimC[num_dim + i];
}
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;
    
    //output 매트릭스 stride는 그냥 dim에 맞게 해라. 그게 맞다,,,, 귀찮게 하지 말고.....

    for (int i = 0; i < (dimA[little_num_dim - 1] + tile_SIZE - 1) / tile_SIZE; i++) {
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdx_A = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
        if (row < dimA[little_num_dim - 2] && (i * tile_SIZE + threadIdx.x) < dimA[little_num_dim - 1])
            s_a[threadIdx.y][threadIdx.x] = A[matIdx_A + row * dimA[2*little_num_dim - 2] + (i * tile_SIZE + threadIdx.x)];
        else 
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        

        if (col < dimB[num_dim-1] && (i * tile_SIZE + threadIdx.y) < dimA[num_dim-1])
            s_b[threadIdx.y][threadIdx.x] = B[matIdx_B  + (i * tile_SIZE + threadIdx.y) * dimB[2*num_dim - 2] + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < dimC[num_dim - 2] && col < dimC[num_dim - 1]) {
        if (bias)
            tmp = tmp + bias[(bias_row?row:col)];
        C[matIdx_C * dimC[2*num_dim - 3] + row * dimC[2*num_dim - 2] + col] = tmp;
    }

}

__global__ void tiledMM_Half_bigA(float *A, float *B, float *C, float *bias, int* A_dim_stride, int *B_dim_stride, int C_mat_dim, int num_dim_A, int num_dim_B, char bias_row){
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    int A_matidx = blockIdx.z;
    
    float tmp = 0.0f;
    //K = A_dim_stride[A->num_dim - 1]
    //M = A_dim_stride[A->num_dim - 2]
    //N = B_dim_stride[B->num_dim - 1]
    for (int i = 0; i < (A_dim_stride[num_dim_A - 1] + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < A_dim_stride[num_dim_A - 2] && (i * tile_SIZE + threadIdx.x) < A_dim_stride[num_dim_A - 1])
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdx_A = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
            s_a[threadIdx.y][threadIdx.x] = A[A_matidx * A_dim_stride[num_dim_A + num_dim_A - 3] + row * A_dim_stride[num_dim_A + num_dim_A - 2] + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < B_dim_stride[num_dim_B - 1] && (i * tile_SIZE + threadIdx.y) < A_dim_stride[num_dim_A - 1])
            s_b[threadIdx.y][threadIdx.x] = B[(i * tile_SIZE + threadIdx.y) * B_dim_stride[num_dim_B + num_dim_B - 2] + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        // if(row==0 && col== 0 && A_matidx == 1){
        // }
        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < A_dim_stride[num_dim_A - 2] && col < B_dim_stride[num_dim_B - 1]) {
        if (bias)
            tmp = tmp + bias[(bias_row?row:col)];
        C[blockIdx.z * C_mat_dim + row * B_dim_stride[num_dim_B - 1] + col] = tmp;
        
    }

}




__global__ void tiledMM_Half_bigB(float *A, float *B, float *C, float *bias, int* A_dim_stride, int *B_dim_stride, int C_mat_dim, int num_dim_A, int num_dim_B, char bias_row){
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    int B_matidx =  blockIdx.z;
    
    float tmp = 0.0f;
    //K = A_dim_stride[A->num_dim - 1]
    //M = A_dim_stride[A->num_dim - 2]
    //N = B_dim_stride[B->num_dim - 1]
    for (int i = 0; i < (A_dim_stride[num_dim_A - 1] + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < A_dim_stride[num_dim_A - 2] && (i * tile_SIZE + threadIdx.x) < A_dim_stride[num_dim_A - 1])
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdx_A = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
            s_a[threadIdx.y][threadIdx.x] = A[row * A_dim_stride[num_dim_A + num_dim_A - 2] + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < B_dim_stride[num_dim_B - 1] && (i * tile_SIZE + threadIdx.y) < A_dim_stride[num_dim_A - 1])
            s_b[threadIdx.y][threadIdx.x] = B[B_matidx * B_dim_stride[num_dim_B + num_dim_B - 3] + (i * tile_SIZE + threadIdx.y) * B_dim_stride[num_dim_B + num_dim_B - 2] + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < A_dim_stride[num_dim_A - 2] && col < B_dim_stride[num_dim_B - 1]) {
        if (bias)
            tmp = tmp + bias[(bias_row?row:col)];
        C[blockIdx.z * C_mat_dim + row * B_dim_stride[num_dim_B - 1] + col] = tmp;
    }

}



__global__ void tiledMM_2d(float *A, float *B, float *C, float *bias, int M, int N, int K, int A_stride, int B_stride, char bias_row) {    
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;

    for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < M && (i * tile_SIZE + threadIdx.x) < K)
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdx_A = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
            s_a[threadIdx.y][threadIdx.x] = A[row * A_stride + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (i * tile_SIZE + threadIdx.y) < K)
            s_b[threadIdx.y][threadIdx.x] = B[(i * tile_SIZE + threadIdx.y) * B_stride + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();

    }

    if (row < M && col < N) {
        if (bias)
            tmp = tmp + bias[(bias_row?row:col)];
        C[row * N + col] = tmp;
    }
}

__global__ void checkmem(float* arr, int len){
    for(int i=0; i < len; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

Tensor* matmul(Tensor* dC, Tensor* dA, Tensor* dB){
    if(!dC||!dB||!dA){
        printf("matmul : one of Tensor is NULL.\n");
        return NULL;
    }
    if(dC->isSub){
        printf("matmul : The result Matrix can not be a sub Matrix.\n");
        return NULL;
    }
    if(dC->device_type != dB->device_type || dB->device_type != dA->device_type){
        printf("matmul : Three matrices are in different device.");
        return NULL;
    }

    
    if(dC->num_dim <2 || dB->num_dim <2 || dA->num_dim <2){
        printf("matmul : One of three Tensor has less than 2 dimensions.(Try to use matmul().)\n");
        return NULL;
    }
    

    //2. matrix col and row has to be the same.
    if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim - 2]){
        printf("matmul : tensor's row and column does not match.(%d %d)\n", dA->dim[dA->num_dim - 1], dB->dim[dB->num_dim - 2]);
        return NULL;
    }

    if(dA->dim[dA->num_dim - 2] != dC->dim[dC->num_dim - 2] || dB->dim[dB->num_dim - 1] != dC->dim[dC->num_dim - 1]){
        printf("matmul : tensor's row and column does not match.(%d %d))\n", dC->dim[dC->num_dim - 2], dC->dim[dC->num_dim - 1]);
        return NULL;
    }
    if(dA->device_type == 0){
        printf("matmul : device type CPU currently not allowed. \n");
        return NULL;
    }


    Tensor* bigTensor, *smallTensor;

    if(dA->num_dim >= dB->num_dim){
        bigTensor = dA;
        smallTensor = dB;
    }else{
        bigTensor = dB;
        smallTensor = dA;
    }


    if(bigTensor->num_dim != dC->num_dim){
        printf("matmul : dC num_dim has to have same num_dim as bigger Tensor.\n");
        return NULL;
    }
    ///////////////////////////////////////////////////////////////////////////////

    if(bigTensor->num_dim ==2){
        cudaSetDevice(bigTensor->device_type-1);
        dim3 dimGrid((dB->dim[dB->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dA->dim[dA->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, 1); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
        dim3 dimBlock(tile_SIZE, tile_SIZE);
        tiledMM_2d<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, NULL, dA->dim[0], dB->dim[1], dA->dim[1], dA->stride[0], dB->stride[0], 0);
    
    }
    else{
        //num_dim이 모두 2보다 클 때
        for(int i=0; i < bigTensor->num_dim - 2; i++){
            if(bigTensor->dim[i] != dC->dim[i] && bigTensor->dim[i] != 1){
                printf("matmul : %d dimension of Tensor(big) does not Match.\n", i);
                return NULL;
            }
        }
        
        if(smallTensor->num_dim == 2){

            cudaSetDevice(dA->device_type - 1);
            dim3 dimGrid((dC->dim[dC->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dC->dim[dC->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, dC->sizeTensor / dC->stride[bigTensor->num_dim - 3]); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
            dim3 dimBlock(tile_SIZE, tile_SIZE);
            if(dA == bigTensor)
                tiledMM_Half_bigA<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, NULL, dA->d_dim_stride, dB->d_dim_stride, dC->stride[dC->num_dim - 3], dA->num_dim, dB->num_dim, 0);
            else
                tiledMM_Half_bigB<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, NULL, dA->d_dim_stride, dB->d_dim_stride, dC->stride[dC->num_dim - 3], dA->num_dim, dB->num_dim, 0);

        }else{

            int cont = bigTensor->num_dim - smallTensor->num_dim;

            //1. The tensor has to match if one of them is not 1.
            for(int i=0; i < smallTensor->num_dim - 2; i++){
                if(smallTensor->dim[i] != dC->dim[i+cont] && smallTensor->dim[i] != 1){
                    printf("matmul : %d dimension of Tensor(small) does not Match.\n", i);
                    return NULL;
                }
            }

            cudaSetDevice(dA->device_type - 1);
            dim3 dimGrid((dC->dim[dC->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dC->dim[dC->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, dC->sizeTensor / dC->stride[dC->num_dim - 3]); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
            dim3 dimBlock(tile_SIZE, tile_SIZE);
            if(dA == bigTensor)
                compTiledMM_Abig<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, NULL, dA->d_dim_stride, dB->d_dim_stride, dC->d_dim_stride, dA->num_dim, dB->num_dim, 0);
            else
                compTiledMM_Bbig<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, NULL, dA->d_dim_stride, dB->d_dim_stride, dC->d_dim_stride, dA->num_dim, dB->num_dim, 0);
            
        }
        
    }
    ///////////////////////////////////////////////////////////////////////////////
    return dC;
    
    //if [5, 1, 3, 4, 5] x [4, 1, 5, 4]
}


Tensor* matmul_bias(Tensor* dC, Tensor* dA, Tensor* dB, Tensor* dbias, char rowwise_bias){
    if(!dC||!dB||!dA||!dbias){
        printf("matmul_bias : one of Tensor is NULL.\n");
        return NULL;
    }
    if(dC->isSub){
        printf("matmul_bias : The result Matrix can not be a sub Matrix.\n");
        return NULL;
    }
    if(dC->device_type != dB->device_type || dB->device_type != dA->device_type || dbias->device_type != dA->device_type){
        printf("matmul_bias : Four matrices are in different device.");
        return NULL;
    }

    
    if(dC->num_dim <2 || dB->num_dim <2 || dA->num_dim <2){
        printf("matmul_bias : One of three Tensor has less than 2 dimensions.(Try to use matmul().)\n");
        return NULL;
    }
    
    if(dbias->num_dim != 1){
        if(dbias->num_dim != 2 || dbias->dim[1] != dC->dim[dC->num_dim - 1]){
            printf("matmul_bias : bias not an appropriate dimension.\n");
            return NULL;
        }
    }

    if(!rowwise_bias && dbias->dim[dbias->num_dim - 1] != dC->dim[dC->num_dim - 1]){
        printf("matmul_bias : Bias size does not fit to size of row(dB's last dim).\n");
        return NULL;
    }
    if(rowwise_bias && dbias->dim[dbias->num_dim - 1] != dC->dim[dC->num_dim - 2]){
        printf("matmul_bias : Bias size does not fit to size of column(dA's second last dim).\n");
        return NULL;
    }
    //2. matrix col and row has to be the same.
    if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim - 2]){
        printf("matmul_bias : tensor's row and column does not match.(%d %d)\n", dA->dim[dA->num_dim - 1], dB->dim[dB->num_dim - 2]);
        return NULL;
    }

    if(dA->dim[dA->num_dim - 2] != dC->dim[dC->num_dim - 2] || dB->dim[dB->num_dim - 1] != dC->dim[dC->num_dim - 1]){
        printf("matmul_bias : tensor's row and column does not match.\n");
        return NULL;
    }

    if(dA->device_type == 0){
        printf("matmul : device type CPU currently not allowed. \n");
        return NULL;
    }
    
    Tensor* bigTensor, *smallTensor;

    if(dA->num_dim >= dB->num_dim){
        bigTensor = dA;
        smallTensor = dB;
    }else{
        bigTensor = dB;
        smallTensor = dA;
    }


    if(bigTensor->num_dim != dC->num_dim){
        printf("matmul_bias : dC num_dim has to have same num_dim as bigger Tensor.\n");
        return NULL;
    }

    if(bigTensor->num_dim ==2){
        cudaSetDevice(bigTensor->device_type-1);
        dim3 dimGrid((dB->dim[dB->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dA->dim[dA->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, 1); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
        dim3 dimBlock(tile_SIZE, tile_SIZE);
        tiledMM_2d<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, dbias->T, dA->dim[0], dB->dim[1], dA->dim[1], dA->stride[0], dB->stride[0], rowwise_bias);
    
    }
    else{
        //num_dim이 모두 2보다 클 때
        for(int i=0; i < bigTensor->num_dim - 2; i++){
            if(bigTensor->dim[i] != dC->dim[i] && bigTensor->dim[i] != 1){
                printf("matmul_bias : %d dimension of Tensor(big) does not Match.\n", i);
                return NULL;
            }
        }
        
        if(smallTensor->num_dim == 2){

            cudaSetDevice(dA->device_type - 1);
            dim3 dimGrid((dC->dim[dC->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dC->dim[dC->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, dC->sizeTensor / dC->stride[bigTensor->num_dim - 3]); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
            dim3 dimBlock(tile_SIZE, tile_SIZE);
            if(dA == bigTensor)
                tiledMM_Half_bigA<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, dbias->T, dA->d_dim_stride, dB->d_dim_stride, dC->stride[dC->num_dim - 3], dA->num_dim, dB->num_dim, rowwise_bias);
            else
                tiledMM_Half_bigB<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, dbias->T, dA->d_dim_stride, dB->d_dim_stride, dC->stride[dC->num_dim - 3], dA->num_dim, dB->num_dim, rowwise_bias);

        }else{

            int cont = bigTensor->num_dim - smallTensor->num_dim;

            //1. The tensor has to match if one of them is not 1.
            for(int i=0; i < smallTensor->num_dim - 2; i++){
                if(smallTensor->dim[i] != dC->dim[i+cont] && smallTensor->dim[i] != 1){
                    printf("matmul_bias : %d dimension of Tensor(small) does not Match.\n", i);
                    return NULL;
                }
            }

            cudaSetDevice(dA->device_type - 1);
            dim3 dimGrid((dC->dim[dC->num_dim - 1] + tile_SIZE - 1) / tile_SIZE, (dC->dim[dC->num_dim - 2] + tile_SIZE - 1) / tile_SIZE, dC->sizeTensor / dC->stride[dC->num_dim - 3]); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
            dim3 dimBlock(tile_SIZE, tile_SIZE);
            if(dA == bigTensor)
                compTiledMM_Abig<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, dbias->T, dA->d_dim_stride, dB->d_dim_stride, dC->d_dim_stride, dA->num_dim, dB->num_dim, rowwise_bias);
            else
                compTiledMM_Bbig<<<dimGrid, dimBlock>>>(dA->T, dB->T, dC->T, dbias->T, dA->d_dim_stride, dB->d_dim_stride, dC->d_dim_stride, dA->num_dim, dB->num_dim, rowwise_bias);
            
        }
        
    }
    return dC;
    
    //if [5, 1, 3, 4, 5] x [4, 1, 5, 4]
    

}










__global__ void ReLU(float* T, int sizeTensor){
    int inx = blockIdx.x * blockDim.x + threadIdx.x;
    if(inx < sizeTensor)
        T[inx] = (T[inx] >=0) ? T[inx] : 0;
}


Tensor* ReLU_inline(Tensor* ten){
    if(!ten){
        printf("ReLU_inline : No Tensor.\n");
        return NULL;
    }
    if(ten->device_type){
        cudaSetDevice(ten->device_type - 1);
        ReLU<<<(ten->sizeTensor+tile_SIZE-1)/tile_SIZE, tile_SIZE*tile_SIZE>>>(ten->T, ten->sizeTensor);
    }else{
        for(int i=0; i < ten->sizeTensor; i++){
            ten->T[i] = (ten->T[i] >=0) ? ten->T[i] : 0;
        }
        
    }
    return ten;
}

__global__ void gelu_(float* in, float* out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {  
        out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}

Tensor* gelu_Tensor(Tensor* ten){

    cudaSetDevice(ten->device_type - 1);
    int s_tile_SIZE = tile_SIZE * tile_SIZE;//no special drawbacks in parallel sequence 임
    gelu_<<< (ten->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(ten->T, ten->T, ten->sizeTensor);
    return ten;
}




__global__ void softmax_(float* dst,float* T, int batch, int label_num){//inx는 batch 크기이다. 
    int inx = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp;
    double clip_value = 80;
    if(inx < batch){
        float max = -__FLT_MAX__;
        float sum = 0;
        for(int i=0; i < label_num; i++){
           max = T[inx * label_num + i];
        }
        for(int i=0; i < label_num; i++){
            tmp = T[inx * label_num + i]-max;
            if(tmp > clip_value)
                tmp = clip_value;
            else if(tmp < -clip_value)
                tmp = -clip_value;
            sum +=expf(tmp);
        }
        for(int i=0; i < label_num; i++){
            tmp = T[inx * label_num + i]-max;
            if(tmp > clip_value)
                tmp = clip_value;
            else if(tmp < -clip_value)
                tmp = -clip_value;
            dst[inx * label_num + i] = expf(tmp)/sum;
        }
    }
    __syncthreads();
}

Tensor* softMax(Tensor* dst, Tensor* src){
    if(!src ||!dst){
        printf("softMax : no Tensor.\n");
        return NULL;
    }
    if(!dst->device_type != !src->device_type){
        printf("sortMax : Two Tensors are in different device.\n");
    }

    if(src->device_type){
        cudaSetDevice(src->device_type - 1);
        int s_tile_SIZE = tile_SIZE * tile_SIZE; //no special drawbacks in parallel sequence 임

        softmax_<<< (dst->dim[0] + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, src->T, src->dim[0], src->dim[1]);
    }else{
        for(int i=0; i < src->dim[0]; i++){
            float max =  -__FLT_MAX__;
            float sum = 0;
            for(int j=0; j < src->dim[1]; j++){
                if(src->T[i * src->stride[0] + j] > max)
                    max = src->T[i * src->stride[0]];
            }
            for(int j=0; j < src->dim[1]; j++){
                sum += expf(src->T[i * src->stride[0] + j]-max);
            }
            for(int j=0; j < src->dim[1]; j++){
                dst->T[i * dst->stride[0] + j] = expf(src->T[i * src->stride[0] + j]-max)/sum;
            }
        }

    }
    return dst;
}



__global__ void elementwise_add_(float* dC, float* dA, float* dB, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        dC[inx] = dA[inx] + dB[inx];
}
__global__ void elementwise_sub_(float* dC, float* dA, float* dB, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        dC[inx] = dA[inx] - dB[inx];
}
__global__ void elementwise_mult_(float* dC, float* dA, float* dB, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        dC[inx] = dA[inx] * dB[inx];
}
__global__ void elementwise_div_(float* dC, float* dA, float* dB, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        dC[inx] = dA[inx] / dB[inx];
}
__global__ void elementwise_mask_(float* dC, float* dA, float* dB, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        dC[inx] = (dB[inx]>0)? dA[inx] : 0;
}

__global__ void elementwise_broadcast_add_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dBig[inx] + dLittle[inx_little];
}
__global__ void elementwise_broadcast_mult_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dBig[inx] * dLittle[inx_little];
}

__global__ void elementwise_broadcast_sub_Abig_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dBig[inx] - dLittle[inx_little];
}

__global__ void elementwise_broadcast_sub_Bbig_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dLittle[inx_little] - dBig[inx];
}

__global__ void elementwise_broadcast_div_Abig_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dBig[inx] / dLittle[inx_little];
}

__global__ void elementwise_broadcast_div_Bbig_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    int inx_little = inx % len_little;
    if(inx < len_Big)
        dC[inx] = dLittle[inx_little] / dBig[inx];
}

Tensor* elementWise_Tensor(Tensor* dC, Tensor* dA, char operand, Tensor* dB){ 
    if(!dC || !dA || !dB){
        printf("elementWise_Tensor : no Tensor\n");
        return NULL;
    }
    if(dC->device_type != dB->device_type || dB->device_type != dA->device_type){
        printf("elementWise_Tensor : Device type does not match\n");
        return NULL;
    }
    
    if(dC->num_dim != dB->num_dim && dC->num_dim != dA->num_dim){
        printf("elementWise_Tensor : Number of dim does not match.\n");
        return NULL;
    }

    // Handle broadcasting if num_dim is different between dA and dB
    if(dA->num_dim != dB->num_dim){
        Tensor* bigTensor, *littleTensor;

        if(dA->num_dim > dB->num_dim){
            bigTensor = dA;
            littleTensor = dB;
        }else{
            bigTensor = dB;
            littleTensor = dA;
        }
        int cont = bigTensor->num_dim - littleTensor->num_dim;
        for(int i=0; i < littleTensor->num_dim; i++){
            if(bigTensor->dim[i+cont] != littleTensor->dim[i]){
                printf("elementWise_Tensor : dimension does not match.\n");
                return NULL;
            }
        }

        // GPU computation case
        if(dC->device_type){
            cudaSetDevice(dC->device_type - 1);
            int s_tile_SIZE = tile_SIZE * tile_SIZE;

            // Handle different operators on GPU
            if(operand == '+'){                                                                             //elementwise_broadcast_add_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little)
                elementwise_broadcast_add_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
            }
            else if(operand == '-'){
                if(bigTensor == dA)
                    elementwise_broadcast_sub_Abig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
                else{
                    elementwise_broadcast_sub_Bbig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
                }
            }
            else if(operand == '*'){
                elementwise_broadcast_mult_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
            }
            else if(operand == '/'){
                if(bigTensor == dA)
                    elementwise_broadcast_div_Abig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
                else{
                    elementwise_broadcast_div_Bbig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, bigTensor->T, littleTensor->T, bigTensor->sizeTensor, littleTensor->sizeTensor);
                }
            }
            else if(operand == 'M' || operand == 'm'){
                elementwise_mask_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            } else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }

        } else {  // CPU computation case
            int i_little = 0;
            if(operand == '+'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    dC->T[i] = bigTensor->T[i] + littleTensor->T[i_little];
                }
            }
            else if(operand == '-'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    dC->T[i] = bigTensor->T[i] - littleTensor->T[i_little];
                }
            }
            else if(operand == '*'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    dC->T[i] = bigTensor->T[i] * littleTensor->T[i_little];
                }
            }
            else if(operand == '/'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    dC->T[i] = bigTensor->T[i] / littleTensor->T[i_little];
                }
            }
            else if(operand == 'M' || operand == 'm'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    dC->T[i] = (littleTensor->T[i_little]>0) ? bigTensor->T[i] : 0;
                }
            }
            else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }
        }

    } else {
        // Dimensions match, proceed with element-wise operations
        for(int i=0; i < dC->num_dim; i++){
            if(dC->dim[i] != dB->dim[i] || dC->dim[i] != dA->dim[i]){
                printf("elementWise_Tensor : dimension does not match.\n");
                return NULL;
            }
        }

        // GPU computation case
        if(dC->device_type){
            cudaSetDevice(dC->device_type - 1);
            int s_tile_SIZE = tile_SIZE * tile_SIZE;

            if(operand == '+'){
                elementwise_add_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            }
            else if(operand == '-'){
                elementwise_sub_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            }
            else if(operand == '*'){
                elementwise_mult_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            }
            else if(operand == '/'){
                elementwise_div_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            }
            else if(operand == 'M' || operand == 'm'){
                elementwise_mask_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dC->T, dA->T, dB->T, dC->sizeTensor);
            } else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }

        } else {  // CPU computation case
            if(operand == '+'){
                for(int i=0; i < dC->sizeTensor; i++){
                    dC->T[i] = dA->T[i] + dB->T[i];
                }
            }
            else if(operand == '-'){
                for(int i=0; i < dC->sizeTensor; i++){
                    dC->T[i] = dA->T[i] - dB->T[i];
                }
            }
            else if(operand == '*'){
                for(int i=0; i < dC->sizeTensor; i++){
                    dC->T[i] = dA->T[i] * dB->T[i];
                }
            }
            else if(operand == '/'){
                for(int i=0; i < dC->sizeTensor; i++){
                    dC->T[i] = dA->T[i] / dB->T[i];
                }
            }
            else if(operand == 'M' || operand == 'm'){
                for(int i=0; i < dC->sizeTensor; i++){
                    dC->T[i] = dB->T[i] ? dA->T[i] : 0;
                }
            }
            else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }
        }
    }

    return dC;
}


__global__ void rowwise_sum_(float* dst, float* src, int row, int col){// 모든 row를 다 더한다. ->col 의 개수만큼의 inx
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < col){
        float tmp = 0;
        for(int i=0; i < row; i++){      
            tmp += src[col * i + inx];
        }
        dst[inx] = tmp;
    }
}

__global__ void colwise_sum_(float* dst, float* src, int row, int col){// 모든 column을 다 더한다. -> row의 개수만큼의 inx
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < row){
        float tmp = 0;
        for(int i=0; i < col; i++){
            tmp += src[col * inx +i];
        }
        dst[inx] = tmp;
    }
}


Tensor* rowcolwise_sum(Tensor*dst, Tensor*src, char axis){
    if(!dst|| !src){
        printf("rowcolwise_sum : no Tensor.\n");
        return NULL;
    }
    if(dst->device_type != src->device_type){
        printf("rowcolwise_sum : Tensors on different device.\n");
        return NULL;
    }
    if(dst->isSub || src->isSub){
        printf("rowcolwise_sum : This function can not have subTensor.\n");
    }
    if(dst->device_type){
        cudaSetDevice(dst->device_type - 1);
        int s_tile_SIZE = tile_SIZE * tile_SIZE;
        if(axis == 0){//rowwise
            if(src->dim[src->num_dim-1] != dst->sizeTensor){
                printf("rowcolwise_sum : row length and size of dst must have same length.\n");
                return NULL;
            }
            rowwise_sum_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, src->T, src->dim[src->num_dim - 2], dst->sizeTensor);
        }else if(axis == 1){
            if(src->dim[src->num_dim-2] != dst->sizeTensor){
                printf("rowcolwise_sum : col length and size of dst must have same length.\n");
                return NULL;
            }
            colwise_sum_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, src->T, dst->sizeTensor, src->dim[src->num_dim-1]);
        }
    }else{
        if(axis == 0){
            if(src->dim[src->num_dim-1] != dst->sizeTensor){
                printf("rowcolwise_sum : row length and size of dst must have same length.\n");
                return NULL;
            }
            for(int j=0; j < src->dim[src->num_dim-1];j++){
                dst->T[j] = 0;
                for(int i=0; i < src->dim[src->num_dim-2];i++){
                    dst->T[j] += src->T[i *src->dim[src->num_dim-1]+ j];
                }
            }
        }else if(axis==1){
            if(src->dim[src->num_dim-2] != dst->sizeTensor){
                printf("rowcolwise_sum : col length and size of dst must have same length.\n");
                return NULL;
            }
            for(int i=0; i < src->dim[src->num_dim-2];i++){
                dst->T[i] = 0;
                for(int j=0; j < src->dim[src->num_dim-1];j++){
                    dst->T[i] += src->T[i *src->dim[src->num_dim-1]+ j];
                }
            }
        }
    }
    return dst;
}

__global__ void scalar_Tensor_sum_(float *T, float scalar, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len){
        T[inx] = T[inx] + scalar;
    }
}
__global__ void scalar_Tensor_sub_(float *T, float scalar, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len){
        T[inx] = -T[inx] + scalar;
    }
}
__global__ void scalar_Tensor_mult_(float *T, float scalar, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len){
        T[inx] = T[inx] * scalar;
    }
}
Tensor* scalar_Tensor(Tensor*dst, char operand ,float scalar){
    if(!dst){
        printf("no tensor\n");
        return NULL;
    }
    if(dst->device_type){
        int s_tile_SIZE = tile_SIZE * tile_SIZE;
        cudaSetDevice(dst->device_type-1);
        if(operand == '+'){
            scalar_Tensor_sum_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, scalar, dst->sizeTensor);
        }else if(operand == '-'){
            scalar_Tensor_sub_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, scalar, dst->sizeTensor);
        }else if(operand == '*'){
            scalar_Tensor_mult_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, scalar, dst->sizeTensor);
        }else{
            printf("scalar_Tensor : not an appropriate operand\n");
        }

    }else{
        int* tmp_Inx = (int*)malloc(sizeof(int) * dst->num_dim);//=
        for(int i=0; i < dst->num_dim; i++){
            tmp_Inx[i] = 0;
        }
        int inx;
        if(operand == '+'){           
            while(tmp_Inx[0] < dst->dim[0]){
                inx = 0;
                for(int i=0; i < dst->num_dim; i++){
                    inx += tmp_Inx[i]*dst->stride[i];
                }

                dst->T[inx] += scalar;

                tmp_Inx[dst->num_dim - 1]++;
                for(int i= dst->num_dim - 1; i > 0; i--){
                    if(tmp_Inx[i] >= dst->dim[i]){
                        tmp_Inx[i-1]++;
                        tmp_Inx[i] = 0;
                    }
                }
            }
        }else if(operand == '-'){
            while(tmp_Inx[0] < dst->dim[0]){
                inx = 0;
                for(int i=0; i < dst->num_dim; i++){
                    inx += tmp_Inx[i]*dst->stride[i];
                }

                dst->T[inx] = scalar - dst->T[inx];

                tmp_Inx[dst->num_dim - 1]++;
                for(int i= dst->num_dim - 1; i > 0; i--){
                    if(tmp_Inx[i] >= dst->dim[i]){
                        tmp_Inx[i-1]++;
                        tmp_Inx[i] = 0;
                    }
                }
            }
        }else if(operand == '*'){
            while(tmp_Inx[0] < dst->dim[0]){
                inx = 0;
                for(int i=0; i < dst->num_dim; i++){
                    inx += tmp_Inx[i]*dst->stride[i];
                }

                dst->T[inx] *= scalar;

                tmp_Inx[dst->num_dim - 1]++;
                for(int i= dst->num_dim - 1; i > 0; i--){
                    if(tmp_Inx[i] >= dst->dim[i]){
                        tmp_Inx[i-1]++;
                        tmp_Inx[i] = 0;
                    }
                }
            }
        }else{
            printf("scalar_Tensor : not an appropriate operand\n");
        }
        free(tmp_Inx);//=
    }
    return dst;
}

__global__ void normalize_(float* output, float* input, int layer_size/*layer*/,int layer_num/*num_of_layer*/, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;// compute each layer

    if (idx < layer_num) {
        float mean = 0.0f;
        float variance = 0.0f;
        
        for (int i = 0; i < layer_size; i++) {
            mean += input[idx * layer_size + i];
        }
        mean /= layer_size;

        for (int i = 0; i < layer_size; i++) {
            variance += (input[idx * layer_size + i] - mean) * (input[idx * layer_size + i] - mean);
        }
        
        variance /= layer_size;
        
        for(int i=0; i < layer_size; i++)
            output[idx * layer_size + i] = (input[idx * layer_size + i] - mean) / sqrtf(variance + epsilon);
    }
    __syncthreads();
}


Tensor* normalize(Tensor* dst, Tensor* src){
    if(!dst||!src){
        printf("no Tensor.\n");
        return NULL;
    }

    if(dst->num_dim != src->num_dim){
        printf("two tensor has different shape.\n");
        return NULL;
    }

    normalize_<<<((dst->sizeTensor/dst->stride[dst->num_dim - 2] + tile_SIZE - 1)/tile_SIZE), tile_SIZE>>>(dst->T, src->T, dst->stride[dst->num_dim - 2], dst->sizeTensor/dst->stride[dst->num_dim - 2], 1e-06);
    // normalize_<<<((dst->sizeTensor/dst->stride[dst->num_dim - 2] + tile_SIZE - 1)/tile_SIZE), tile_SIZE>>>(dst->T, src->T, 5, 2, 1e-06);
    return dst;
}



Tensor* addGrad(Tensor* ten){

    if(ten->dT){
        printf("addGrad : Tensor grad already allocated.\n");
        return NULL;
    }
    if(!ten->device_type){
        ten->dT = (float*)malloc(ten->sizeTensor * sizeof(float));
    }else{
        cudaSetDevice(ten->device_type-1);
        cudaMalloc(&ten->dT, ten->sizeTensor * sizeof(float));
    }
    return ten;
}




Tensor* matmul_grad(Tensor* dC,char dC_Grad, Tensor* dA,char dA_Grad, Tensor* dB,char dB_Grad){
 
    /////////////////////////////////grad change//////////////////////////////////////////////
    float *Ts[3];
    if(dA_Grad == GRAD_TRUE){
        Ts[0] = dA->T;
        if(!dA->dT){
            printf("matmul_grad : dA has no grad.\n");
            return NULL;
        }
        dA->T = dA->dT;
    }
    if(dB_Grad&& dA != dB){
        Ts[1] = dB->T;
        if(!dB->dT){
            printf("matmul_grad : dB has no grad.\n");
            return NULL;
        }
        dB->T = dB->dT;
    }
    if(dC_Grad){
        Ts[2] = dC->T;
        if(!dC->dT){
            printf("matmul_grad : dC has no grad.\n");
            return NULL;
        }
        dC->T = dC->dT;
    }
    matmul(dC, dA, dB);
    //////////////////////////////////grad 복구/////////////////////////////////////////////

    if(dA_Grad){
        dA->dT = dA->T;
        dA->T = Ts[0];
    }
    if(dB_Grad&& dA != dB){
        dB->dT = dB->T;
        dB->T = Ts[1];
    }
    if(dC_Grad){
        dC->dT = dC->T;
        dC->T = Ts[2];
    }
    return dC;   
}


//Tensor* dC, Tensor* dA, char operand, Tensor* dB
Tensor* elementWise_Tensor_grad(Tensor*dC,char dC_Grad, Tensor* dA,char dA_Grad, char operand,Tensor* dB, char dB_Grad){
    
    /////////////////////////////////grad change//////////////////////////////////////////////
    float *Ts[3];
    if(dA_Grad == GRAD_TRUE){
        Ts[0] = dA->T;
        if(!dA->dT){
            printf("elementWise_Tensor_grad : dA has no grad.\n");
            return NULL;
        }
        dA->T = dA->dT;
    }
    if(dB_Grad){
        Ts[1] = dB->T;
        if(!dB->dT){
            printf("elementWise_Tensor_grad : dB has no grad.\n");
            return NULL;
        }
        dB->T = dB->dT;
    }
    if(dC_Grad && !((dA == dC) && dA_Grad) && !((dB == dC) && dB_Grad)){
        Ts[2] = dC->T;
        if(!dC->dT){
            printf("elementWise_Tensor_grad : dC has no grad.\n");
            return NULL;
        }
        dC->T = dC->dT;
    }
    elementWise_Tensor(dC, dA, operand, dB);

    //////////////////////////////////grad 복구/////////////////////////////////////////////

    if(dA_Grad){
        dA->dT = dA->T;
        dA->T = Ts[0];
    }
    if(dB_Grad){
        dB->dT = dB->T;
        dB->T = Ts[1];
    }
    if(dC_Grad && !((dA == dC) && dA_Grad) && !((dB == dC) && dB_Grad)){
        dC->dT = dC->T;
        dC->T = Ts[2];
    }
    return dC;
}

//Tensor* dC, Tensor* dA, char operand, Tensor* dB
Tensor* elementWise_Tensor_grad_2(Tensor*dC,char dC_Grad, Tensor* dA,char dA_Grad, char operand,Tensor* dB, char dB_Grad){
    
    /////////////////////////////////grad change//////////////////////////////////////////////
    float *Ts[3];
    if(dA_Grad == GRAD_TRUE){
        if(!dA->dT){
            printf("elementWise_Tensor_grad : dA has no grad.\n");
            return NULL;
        }
        Ts[0] = dA->dT;
    }else{
        Ts[0] = dA->T;
    }

    if(dB_Grad){
        if(!dB->dT){
            printf("elementWise_Tensor_grad : dB has no grad.\n");
            return NULL;
        }
        Ts[1] = dB->dT;
    }else{
        Ts[1] = dB->T;
    }
    if(dC_Grad){
        if(!dC->dT){
            printf("elementWise_Tensor_grad : dC has no grad.\n");
            return NULL;
        }
        Ts[2] = dC->dT;
    }else{
        Ts[2] = dC->T;
    }

    if(!dC || !dA || !dB){
        printf("elementWise_Tensor : no Tensor\n");
        return NULL;
    }
    if(dC->device_type != dB->device_type || dB->device_type != dA->device_type){
        printf("elementWise_Tensor : Device type does not match\n");
        return NULL;
    }
    
    if(dC->num_dim != dB->num_dim && dC->num_dim != dA->num_dim){
        printf("elementWise_Tensor : Number of dim does not match.\n");
        return NULL;
    }

    // Handle broadcasting if num_dim is different between dA and dB
    if(dA->num_dim != dB->num_dim){
        Tensor* bigTensor, *littleTensor;

        if(dA->num_dim > dB->num_dim){
            bigTensor = dA;
            littleTensor = dB;
        }else{
            bigTensor = dB;
            littleTensor = dA;
            float* tmp = Ts[1];
            Ts[1] = Ts[0];
            Ts[0] = tmp;
        }
        int cont = bigTensor->num_dim - littleTensor->num_dim;
        for(int i=0; i < littleTensor->num_dim; i++){
            if(bigTensor->dim[i+cont] != littleTensor->dim[i]){
                printf("elementWise_Tensor : dimension does not match.\n");
                return NULL;
            }
        }

        // GPU computation case
        if(dC->device_type){
            cudaSetDevice(dC->device_type - 1);
            int s_tile_SIZE = tile_SIZE * tile_SIZE;

            // Handle different operators on GPU
            if(operand == '+'){                                                                             //elementwise_broadcast_add_(float* dC, float* dBig, float* dLittle, int len_Big, int len_little)
                elementwise_broadcast_add_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
            }
            else if(operand == '-'){
                if(bigTensor == dA)
                    elementwise_broadcast_sub_Abig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
                else{
                    elementwise_broadcast_sub_Bbig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
                }
            }
            else if(operand == '*'){
                elementwise_broadcast_mult_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
            }
            else if(operand == '/'){
                if(bigTensor == dA)
                    elementwise_broadcast_div_Abig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
                else{
                    elementwise_broadcast_div_Bbig_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], bigTensor->sizeTensor, littleTensor->sizeTensor);
                }
            }
            else if(operand == 'M' || operand == 'm'){
                elementwise_mask_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            } else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }

        } else {  // CPU computation case
            int i_little = 0;
            if(operand == '+'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    Ts[2][i] = Ts[0][i] + Ts[1][i_little];
                }
            }
            else if(operand == '-'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    Ts[2][i] = Ts[0][i] - Ts[1][i_little];
                }
            }
            else if(operand == '*'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    Ts[2][i] = Ts[0][i] * Ts[1][i_little];
                }
            }
            else if(operand == '/'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    Ts[2][i] = Ts[0][i] / Ts[1][i_little];
                }
            }
            else if(operand == 'M' || operand == 'm'){
                for(int i=0; i < dC->sizeTensor; i++){
                    i_little = i % littleTensor->sizeTensor;
                    Ts[2][i] = (Ts[1][i_little]>0) ? Ts[0][i] : 0;
                }
            }
            else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }
        }

    } else {
        // Dimensions match, proceed with element-wise operations
        for(int i=0; i < dC->num_dim; i++){
            if(dC->dim[i] != dB->dim[i] || dC->dim[i] != dA->dim[i]){
                printf("elementWise_Tensor : dimension does not match.\n");
                return NULL;
            }
        }

        // GPU computation case
        if(dC->device_type){
            cudaSetDevice(dC->device_type - 1);
            int s_tile_SIZE = tile_SIZE * tile_SIZE;

            if(operand == '+'){
                elementwise_add_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            }
            else if(operand == '-'){
                elementwise_sub_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            }
            else if(operand == '*'){
                elementwise_mult_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            }
            else if(operand == '/'){
                elementwise_div_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            }
            else if(operand == 'M' || operand == 'm'){
                elementwise_mask_<<< (dC->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(Ts[2], Ts[0], Ts[1], dC->sizeTensor);
            } else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }

        } else {  // CPU computation case
            if(operand == '+'){
                for(int i=0; i < dC->sizeTensor; i++){
                    Ts[2][i] = Ts[0][i] + Ts[1][i];
                }
            }
            else if(operand == '-'){
                for(int i=0; i < dC->sizeTensor; i++){
                    Ts[2][i] = Ts[0][i] - Ts[1][i];
                }
            }
            else if(operand == '*'){
                for(int i=0; i < dC->sizeTensor; i++){
                    Ts[2][i] = Ts[0][i] * Ts[1][i];
                }
            }
            else if(operand == '/'){
                for(int i=0; i < dC->sizeTensor; i++){
                    Ts[2][i] = Ts[0][i] / Ts[1][i];
                }
            }
            else if(operand == 'M' || operand == 'm'){
                for(int i=0; i < dC->sizeTensor; i++){
                    Ts[2][i] = Ts[1][i] ? Ts[0][i] : 0;
                }
            }
            else {
                printf("elementWise_Tensor : not an appropriate operand\n");
                return NULL;
            }
        }
    }

    return dC;

    //////////////////////////////////grad 복구/////////////////////////////////////////////

    // if(dA_Grad){
    //     dA->dT = dA->T;
    //     dA->T = Ts[0];
    // }
    // if(dB_Grad){
    //     dB->dT = dB->T;
    //     dB->T = Ts[1];
    // }
    // if(dC_Grad && !((dA == dC) && dA_Grad) && !((dB == dC) && dB_Grad)){
    //     dC->dT = dC->T;
    //     dC->T = Ts[2];
    // }
    // return dC;
}



Tensor* rowcolwise_sum_grad(Tensor*dst,int dst_Grad, Tensor*src,int src_Grad, char axis){
    float *Ts[2];
    if(dst_Grad == GRAD_TRUE){
        Ts[0] = dst->T;
        if(!dst->dT){
            printf("rowcolwise_sum_grad : dst has no grad.\n");
            return NULL;
        }
        dst->T = dst->dT;
    }
    if(src_Grad == GRAD_TRUE){
        Ts[1] = src->T;
        if(!src->dT){
            printf("rowcolwise_sum_grad : src has no grad.\n");
            return NULL;
        }
        src->T = src->dT;
    }
    rowcolwise_sum(dst, src, axis);
    if(dst_Grad == GRAD_TRUE){
        dst->T = Ts[0];
    }
    if(src_Grad == GRAD_TRUE){
        src->T = Ts[1];
    }
    return dst;
}


Tensor* copyTensor_grad(Tensor *dst, Tensor *src){
    if(!dst || !src){
        printf("copyTensor : No dst or src\n");
        return NULL;
    }
    if(dst->isSub||src->isSub){
        printf("copyTensor : dst or src is subTensor.");
        return NULL;
    }
    if(dst->num_dim != src->num_dim){
        printf("copyTensor : shape of dst and src doesn't match.\n");
        return NULL;
    }
    for(int i=0; i < dst->num_dim; i++){
        if(dst->dim[i] != src->dim[i]){
            printf("copyTensor : shape of dst and src doesn't match.\n");
            return NULL;
        }
    }

    if(!src->dT){
        printf("src has no dT.\n");
        return NULL;
    }
    if(!dst->device_type && !src->device_type){ //CPU to CPU
        for(int i=0; i < dst->dim[0]*dst->stride[0]; i++)
            dst->T[i] = src->T[i];

    }

    else if(dst->device_type && src->device_type){
        cudaSetDevice(dst->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
        if(!dst->dT){
            cudaMalloc(&dst->dT, sizeof(float) * dst->sizeTensor);
        }
        cudaMemcpy(dst->dT, src->dT, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);

    }
    else if(dst->device_type){
        cudaSetDevice(dst->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyHostToDevice);
        if(!dst->dT){
            cudaMalloc(&dst->dT, sizeof(float) * dst->sizeTensor);
        }
        cudaMemcpy(dst->dT, src->dT, dst->sizeTensor * sizeof(float), cudaMemcpyHostToDevice);
    }else{
        cudaSetDevice(src->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToHost);
        if(!dst->dT){
            dst->dT = (float*)malloc(sizeof(float) * dst->sizeTensor);
        }
        cudaMemcpy(dst->dT, src->dT, dst->sizeTensor * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return dst;
}



Tensor* printTensor_grad(Tensor *ten){
    if(!ten){
        printf("printTensor : No tensor.\n");
        return NULL;
    }
    if(ten->device_type){
        printf("printTensor : GPU mem can not be printed\n");
        return ten;
    }
    infoTensor(ten);
    //==============if ten->num_dim < 3========================
    if(ten->num_dim == 1){
        printf("[ %d ]\n", ten->dim[0]);
        for(int i=0; i < ten->dim[0]; i+=ten->stride[0]){
            printf("%.03f\t", ten->dT[i]);
        }
        printf("\n");
        return ten;
    }

    if(ten->num_dim == 2){
        printf("[ %d %d ]\n", ten->dim[0], ten->dim[1]);
        for(int i=0; i < ten->dim[0]*ten->stride[0]; i+=ten->stride[0]){
            for(int j=0; j < ten->dim[1]*ten->stride[1]; j+=ten->stride[1]){
                printf("%.03f\t", ten->dT[i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }
        printf("\n");
        return ten;
    }
    //=================else====================================
    printf("=\n");

    int* tmp_Inx = (int*)malloc(sizeof(int) * (ten->num_dim - 2));
    for(int i=0; i < ten->num_dim - 2; i++){
        tmp_Inx[i] = 0;
    }
    int inx;
    while(tmp_Inx[0] < ten->dim[0]){
        inx = 0;
        printf("[ ");
        for(int i=0; i < ten->num_dim-2;i++){
            printf("%d ", tmp_Inx[i]);
            inx += tmp_Inx[i] * ten->stride[i];
        }
        printf("- - ]\n");

        for(int i=0; i < ten->dim[ten->num_dim-2]*ten->stride[ten->num_dim-2]; i+=ten->stride[ten->num_dim-2]){
            for(int j=0; j < ten->dim[ten->num_dim-1]*ten->stride[ten->num_dim-1]; j+=ten->stride[ten->num_dim-1]){
                printf("%.03f\t", ten->dT[inx + i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }

        tmp_Inx[ten->num_dim - 3]++;
        for(int i = ten->num_dim - 3; i > 0; i--){
            if(tmp_Inx[i] >= ten->dim[i]){
                tmp_Inx[i-1]++;
                tmp_Inx[i] = 0;
            }
        }
    }
    printf("=\n");
    free(tmp_Inx);
    return ten;
}









__global__ void reset_Tensor_(float* A, int len, int num){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        A[i] = num;
    }
}


Tensor* reset_Tensor(Tensor* dst, int num){
    if(!dst){
        printf("reset_Tensor : no Tensor\n");
        return NULL;
    }
    reset_Tensor_<<<((dst->sizeTensor + tile_SIZE - 1)/tile_SIZE), tile_SIZE>>>(dst->T, dst->sizeTensor, num);
    return dst;
}


__global__ void update_add_(float* T, float* dT, float lr, int len){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if(inx < len)
        T[inx] -= lr * dT[inx];
}

Tensor* updateTensor(Tensor* ten, float learning_rate){
    cudaSetDevice(ten->device_type - 1);
    int s_tile_SIZE = tile_SIZE * tile_SIZE;
    if(ten->dT)
        update_add_<<< (ten->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(ten->T, ten->dT, learning_rate, ten->sizeTensor);
    cudaMemset(ten->dT, 0, ten->sizeTensor * sizeof(float));
    return ten;
}
