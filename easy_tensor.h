#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 16
#define MAX_NUM_DIM 30
#define GRAD_TRUE 1
#define GRAD_FALSE 0
typedef struct Tensor{
    float *T;           //Remember, this is a pointer, not an array. (CPU pointer if device_type == 0, GPU pointer if device_type > 0)
    float *dT;          //derivative of T. (Also CPU pointer if device_type == 0, GPU pointer if device_type > 0)
    int *dim;           //dimension, logical dimension which means memory jump could be different(esp. in subTensor) and that's why we need stride.
    int *stride;        //stride, which tells us how many memory jump we need to make in order to change dim.
    int *d_dim_stride;  //concat(dim, stride) for GPU.
    int num_dim;        //dim, stride, d_dim_s~ length
    int sizeTensor;     //physical size of whole tensor
    char device_type;   //device type(cpu, gpu number)
    char isSub;         //is it subTensor???
    char isParam;       //Tells if this is a parameter(in case if it needs optimization from its derv.)
}Tensor;

Tensor *mallocTensor(int *dim, int num_dim, int device_type);

//makeTensor is to make tensor from string. This way I can allocate tensors more intuitively.
Tensor *makeTensor(const char dim[], int device_type);

//makeTensorbyShape is to make tensor from another tensor. This way I can allocate tensors more intuitively.
Tensor *makeTensorbyShape(Tensor* src, int device_type);

//makeSubTensor is to make subTensor from existing tensor.
Tensor *makeSubTensor(Tensor* src, const char* start_point, const char* dim);

//freeTensor is free...tensor.
void freeTensor(Tensor *ten);


//copyTensor. Should not be subTensor. 
Tensor* copyTensor(Tensor* dst, Tensor* src);

//copy gradient of Tensor this also should not be subTensor.
Tensor* copyTensor_grad(Tensor *dst, Tensor *src);

//copyTransposeTensor. Should not be subTensor.
Tensor* copyTransposeTensor(Tensor* dst, Tensor* src);

//for partial copy(like subTensor), reshape can help you do it.
//"copyReshapeTensor(makeTensor(subDim,num_dim, 1), makeSubTensor(src, sp, subDim, num_dim));"We can copy like this when we want to copy subTensor.
//when tensor is 2dim, it uses transpose function for better performance.
Tensor* copyReshapeTensor(Tensor* dst, Tensor* src, int* reshape);



//값 출력. CPU만 된다.
Tensor* printTensor(Tensor *ten);
Tensor* printTensor_grad(Tensor *ten);


//정보 출력. 다 된다.
Tensor* infoTensor(Tensor *ten);


Tensor* matmul(Tensor* dC, Tensor *dA, Tensor* dB);
Tensor* matmul_bias(Tensor* dC, Tensor* dA, Tensor* dB, Tensor* dbias, char rowwise_bias);
Tensor* matmul_grad(Tensor* dC,char dC_Grad, Tensor* dA,char dA_Grad, Tensor* dB,char dB_Grad);

Tensor* ReLU_inline(Tensor *ten);

Tensor* gelu_Tensor(Tensor* ten);
Tensor* softMax(Tensor* dst, Tensor*src);

//0 : add, 1 : subtract, 2: multiply, 10: mask (dA source, dB mask)
Tensor* elementWise_Tensor(Tensor* dC, Tensor* dA, char operand,Tensor* dB);

Tensor* rowcolwise_sum(Tensor*dst, Tensor*src, char axis);

Tensor* scalar_Tensor(Tensor*dst,char operand ,float scalar);

Tensor* normalize(Tensor*dst, Tensor* src);

Tensor* addGrad(Tensor* ten);

Tensor* reset_Tensor(Tensor* dst, int num);

Tensor* elementWise_Tensor_grad(Tensor*dC, char dC_Grad, Tensor* dA,char dA_Grad, char operand,Tensor* dB, char dB_Grad);
Tensor* elementWise_Tensor_grad_2(Tensor*dC,char dC_Grad, Tensor* dA,char dA_Grad, char operand,Tensor* dB, char dB_Grad);

Tensor* rowcolwise_sum_grad(Tensor*dst,int dst_Grad, Tensor*src,int src_Grad, char axis);

Tensor* updateTensor(Tensor* ten, float learning_rate);
Tensor* updateTensor(Tensor* ten, float learning_rate);

// Helper Functions
void print_progress(int count, int max, float acc);
int accuracy_CPU(Tensor* O, Tensor* Y);
Tensor* copyTensorfromFILE(Tensor* dst, const char* file_name);

// Data Loader
FILE* LoaderINIT(const char* file_name);
Tensor* LoaderNEXT(Tensor* dst, FILE*file);
void LoaderCLOSE(FILE* file);

// Loss
float CrossEntropyLoss(Tensor* CPU_O, Tensor* CPU_Y);
Tensor* CESoftmax_deriv(Tensor* d_der_O, Tensor*d_O, Tensor* d_Y);

#endif // TENSOR_H

//주의할 점. 
//어차피 할당된 곳을 계속 쓰게 되어있다. 굳이 free할 일이 거의 없으므로 웬만하면 inline 또는 dst, src 꼴로 만들어 주는 것이 제일 좋다.
/////////////////////////////////MHA_BLK////////////////////////////////////////////
