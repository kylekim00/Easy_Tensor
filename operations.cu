#include "easy_tensor.h"
#include <string>
#include <iostream>

using namespace std;

class Operation{
public:
    string op_name;
    Tensor** upstream;
    Tensor** downstream;
    int upstream_len;
    int downstream_len;
    Operation(int upstream_len, int downstream_len){
        this->op_name = "Operation";
        this->upstream = (Tensor**)malloc(sizeof(Tensor*) * upstream_len);
        this->downstream = (Tensor**)malloc(sizeof(Tensor*) * downstream_len);
        this->upstream_len = upstream_len;
        this->downstream_len = downstream_len;
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual string getOpname(){
        return op_name;
    }
    virtual ~Operation(){
        free(this->upstream);
        free(this->downstream);
    }
};


class Matmul_OP:public Operation{
public:
    Matmul_OP(Tensor* Y, Tensor* X1, Tensor* X2) : Operation(1, 2){
        op_name = "matmul";
        upstream[0] = Y;
        downstream[0] = X1;
        downstream[1] = X2;
    }
    void forward() override{
        matmul(upstream[0], downstream[0], downstream[1]);
    }
    void backward() override{
        if(upstream[0]->dT){
            //downstream[0] deriv
            //transpose downstream[1] tmp
            int dim[downstream[1]->num_dim];
            for(int i=0; i < downstream[1]->num_dim - 2; i++){
                dim[i] = downstream[1]->dim[i];
            }
            dim[downstream[1]->num_dim - 2] = downstream[1]->dim[downstream[1]->num_dim - 1];
            dim[downstream[1]->num_dim - 1] = downstream[1]->dim[downstream[1]->num_dim - 2];
            Tensor* down1_T = mallocTensor(dim, downstream[1]->num_dim, downstream[1]->device_type);
            copyTransposeTensor(down1_T, downstream[1]);

            ///Transpose downstream[0] tmp
            int dim2[downstream[0]->num_dim];
            for(int i=0; i < downstream[0]->num_dim - 2; i++){
                dim2[i] = downstream[0]->dim[i];
            }
            dim2[downstream[0]->num_dim - 2] = downstream[0]->dim[downstream[0]->num_dim - 1];
            dim2[downstream[0]->num_dim - 1] = downstream[0]->dim[downstream[0]->num_dim - 2];
            Tensor* down0_T = mallocTensor(dim2, downstream[0]->num_dim, downstream[0]->device_type);
            copyTransposeTensor(down0_T, downstream[0]);

            //derivative가 없다.
            Tensor* down_tmp0 = NULL, *down_tmp1 = NULL;
            if(!downstream[0]->dT){//만약 x2가 deriv가 없다. 
                addGrad(downstream[0]);
            }else{
                down_tmp0 = makeTensorbyShape(downstream[0], downstream[0]->device_type);
                down_tmp0 = copyTensor_grad(down_tmp0, downstream[0]);
                // printTensor_grad(copyTensor_grad(makeTensorbyShape(downstream[0], 0), downstream[0]));
                // printTensor_grad(copyTensor_grad(makeTensorbyShape(down_tmp0, 0), down_tmp0));
            }
            if(!downstream[1]->dT){//만약 x1이 deriv가 없다.
                addGrad(downstream[1]);
            }else{
                down_tmp1 = makeTensorbyShape(downstream[1], downstream[1]->device_type);
                down_tmp1 = copyTensor_grad(down_tmp1, downstream[1]);
            }
            //@X2 = X1^T x @Y
            matmul_grad(downstream[1], 1,
                        down0_T, 0,
                        upstream[0], 1);
            //@X1 = @Y x X2^T
            matmul_grad(downstream[0], 1,
                        upstream[0], 1,
                        down1_T, 0
                        );
            // infoTensor(downstream[1]);
            if(down_tmp0){
                downstream[0] = elementWise_Tensor_grad_2(downstream[0], GRAD_TRUE, downstream[0], GRAD_TRUE,'+' , down_tmp0, GRAD_TRUE);
                freeTensor(down_tmp0);
            }
            if(down_tmp1){
                downstream[1] = elementWise_Tensor_grad_2(downstream[1], GRAD_TRUE, downstream[1], GRAD_TRUE,'+' , down_tmp1, GRAD_TRUE);
                freeTensor(down_tmp1);
            }

            freeTensor(down0_T);
            freeTensor(down1_T);
        }else{
            Tensor*up = makeTensorbyShape(upstream[0], upstream[0]->device_type);
            // 1로 채운 upstream 크기의 Tensor
            if(up->device_type)
                reset_Tensor(up, 1);
            else{
                for(int i=0; i < up->sizeTensor; i++)
                    up->T[i] = 1;
            }
            

            //////////////Transpose downstream////////////////
            ///Transpose downstream[1] tmp
            int dim[downstream[1]->num_dim];
            for(int i=0; i < downstream[1]->num_dim - 2; i++){
                dim[i] = downstream[1]->dim[i];
            }
            dim[downstream[1]->num_dim - 2] = downstream[1]->dim[downstream[1]->num_dim - 1];
            dim[downstream[1]->num_dim - 1] = downstream[1]->dim[downstream[1]->num_dim - 2];
            Tensor* down1_T = mallocTensor(dim, downstream[1]->num_dim, downstream[1]->device_type);
            copyTransposeTensor(down1_T, downstream[1]);

            ///Transpose downstream[0] tmp
            int dim2[downstream[0]->num_dim];
            for(int i=0; i < downstream[0]->num_dim - 2; i++){
                dim2[i] = downstream[0]->dim[i];
            }
            dim2[downstream[0]->num_dim - 2] = downstream[0]->dim[downstream[0]->num_dim - 1];
            dim2[downstream[0]->num_dim - 1] = downstream[0]->dim[downstream[0]->num_dim - 2];
            Tensor* down0_T = mallocTensor(dim2, downstream[0]->num_dim, downstream[0]->device_type);
            copyTransposeTensor(down0_T, downstream[0]);
            // printTensor(copyTensor(makeTensorbyShape(down0_T, 0), down0_T));
            //derivative가 없다.
            Tensor* down_tmp0 = NULL, *down_tmp1 = NULL;
            if(!downstream[0]->dT){//만약 x2가 deriv가 없다. 
                addGrad(downstream[0]);
            }else{
                down_tmp0 = makeTensorbyShape(downstream[0], downstream[0]->device_type);
                down_tmp0 = copyTensor_grad(down_tmp0, downstream[0]);
                // printTensor_grad(copyTensor_grad(makeTensorbyShape(downstream[0], 0), downstream[0]));
                // printTensor_grad(copyTensor_grad(makeTensorbyShape(down_tmp0, 0), down_tmp0));
            }
            if(!downstream[1]->dT){//만약 x1이 deriv가 없다.
                addGrad(downstream[1]);
            }else{
                down_tmp1 = makeTensorbyShape(downstream[1], downstream[1]->device_type);
                down_tmp1 = copyTensor_grad(down_tmp1, downstream[1]);
            }

            //@X2 = X1^T x @Y
            matmul_grad(downstream[1], 1,
                        down0_T, 0,
                        up, 0);
            //@X2 = X1^T x @Y
            matmul_grad(downstream[0], 1,
                        up, 0,
                        down1_T, 0
                        );
            
            if(down_tmp0){
                
                downstream[0] = elementWise_Tensor_grad(downstream[0], GRAD_TRUE, downstream[0], GRAD_TRUE,'+' , down_tmp0, GRAD_TRUE);
                freeTensor(down_tmp0);
            }
            if(down_tmp1){
                downstream[1] = elementWise_Tensor_grad(downstream[1], GRAD_TRUE, downstream[1], GRAD_TRUE,'+' , down_tmp1, GRAD_TRUE);
                freeTensor(down_tmp1);
            }

            freeTensor(down0_T);
            freeTensor(down1_T);

            freeTensor(up);
        }

    }
};


Tensor* makeTensorbyTransposedShape(Tensor* src, int device_type){
    int dim[src->num_dim];
    for(int i=0; i < src->num_dim - 2; i++){
        dim[i] = src->dim[i];
    }
    dim[src->num_dim - 2] = src->dim[src->num_dim - 1];
    dim[src->num_dim - 1] = src->dim[src->num_dim - 2];
    return mallocTensor(dim, src->num_dim, device_type);
}

class Matmul_bias_OP:public Operation{
public:
    Tensor* down0_T;
    Tensor* down1_T;
    Matmul_bias_OP(Tensor* Y, Tensor* X1, Tensor* X2, Tensor* bias) : Operation(1, 3){
        op_name = "matmul_bias";
        upstream[0] = Y;
        downstream[0] = X1;
        downstream[1] = X2;
        downstream[2] = bias;

        down0_T = makeTensorbyTransposedShape(downstream[0], downstream[1]->device_type);
        down1_T = makeTensorbyTransposedShape(downstream[1], downstream[1]->device_type);
    }

    ~Matmul_bias_OP(){
        freeTensor(down0_T);
        freeTensor(down1_T);
    }   

    void forward() override{
        matmul_bias(upstream[0], downstream[0], downstream[1], downstream[2], 0);
    }

    void backward() override{
        if(upstream[0]->dT){
            // Use pre-allocated transpose buffers
            // Transpose X (downstream[0]) -> X^T (down0_T)
            copyTransposeTensor(down0_T, downstream[0]);
            // Transpose W (downstream[1]) -> W^T (down1_T)
            copyTransposeTensor(down1_T, downstream[1]);


            //derivative가 없을 경우 grad 저장공간 할당, 있을 경우 기존 grad를 더해준다. 
            Tensor* down_tmp0 = NULL, *down_tmp1 = NULL, *down_tmp2 = NULL;
            if(!downstream[0]->dT){
                addGrad(downstream[0]);
            }else{
                down_tmp0 = makeTensorbyShape(downstream[0], downstream[0]->device_type);
                down_tmp0 = copyTensor_grad(down_tmp0, downstream[0]);
            }
            if(!downstream[1]->dT){
                addGrad(downstream[1]);
            }else{
                down_tmp1 = makeTensorbyShape(downstream[1], downstream[1]->device_type);
                down_tmp1 = copyTensor_grad(down_tmp1, downstream[1]);
            }
            if(!downstream[2]->dT){
                addGrad(downstream[2]);
            }else{
                down_tmp2 = makeTensorbyShape(downstream[2], downstream[2]->device_type);
                down_tmp2 = copyTensor_grad(down_tmp2, downstream[2]);
            }

            //@X1 = @Y x X2^T (dX = dY * W^T)
            matmul_grad(downstream[0], 1,
                        upstream[0], 1,
                        down1_T, 0
                        );

            //@X2 = X1^T x @Y (dW = X^T * dY)
            matmul_grad(downstream[1], 1,
                        down0_T, 0,
                        upstream[0], 1);
            
            //bias
            rowcolwise_sum_grad(
                        downstream[2], GRAD_TRUE,
                        upstream[0], GRAD_TRUE,
                        0
            );

            // Accumulate gradients if they existed previously
            // FIXED: Removed elementWise_Tensor calls that corrupted T values
            if(down_tmp0){
                downstream[0] = elementWise_Tensor_grad(downstream[0], GRAD_TRUE, downstream[0], GRAD_TRUE,'+' , down_tmp0, GRAD_TRUE);
                freeTensor(down_tmp0);
            }
            if(down_tmp1){
                downstream[1] = elementWise_Tensor_grad(downstream[1], GRAD_TRUE, downstream[1], GRAD_TRUE,'+' , down_tmp1, GRAD_TRUE);
                freeTensor(down_tmp1);
            }
            if(down_tmp2){
                downstream[2] = elementWise_Tensor_grad(downstream[2], GRAD_TRUE, downstream[2], GRAD_TRUE,'+' , down_tmp2, GRAD_TRUE);
                freeTensor(down_tmp2);
            }
        }
    }
};

class ReLU_OP:public Operation{
Tensor* mask;
public:
    ReLU_OP(Tensor* X) : Operation(1, 1){
        op_name = "ReLU";
        upstream[0] = X;
        downstream[0] = X;
        mask = makeTensorbyShape(X, 1);
    }
    void forward() override{
        downstream[0] = ReLU_inline(downstream[0]);
        copyTensor(mask, downstream[0]);
        // printTensor(copyTensor(makeTensorbyShape(mask, 0), mask));//======================================
    }
    
    
    void backward() override{
        //dB로 비교를 한다. 
        // 미분 값이 없으면 미분을 하면 안된다. 
        if(downstream[0]->dT){//만약 x1의 deriv가 없다면 derivative는 없다.
            
            // printTensor_grad(copyTensor_grad(makeTensorbyShape(downstream[0], 0), downstream[0]));
            elementWise_Tensor_grad(downstream[0], GRAD_TRUE, upstream[0], GRAD_TRUE, 'm', mask, GRAD_FALSE);
            // printTensor_grad(copyTensor_grad(makeTensorbyShape(downstream[0], 0), downstream[0]));
            // printTensor(copyTensor(makeTensorbyShape(mask, 0), mask));//=============================
        }
    }
    
    ~ReLU_OP() override{
        freeTensor(mask);
    }

};


class Sum_OP:public Operation{
public:
    Sum_OP(Tensor* Y, Tensor* X):Operation(1,1){
        downstream[0] = X;
        upstream[0] = Y;
    }
    void forward() override{

    }
    void backward() override{
        if(!downstream[0]->dT){
            addGrad(downstream[0]);
            float* tmp = downstream[0]->T;
            downstream[0]->T = downstream[0]->dT;
            reset_Tensor(downstream[0], 1);
            downstream[0]->dT = downstream[0]->T;
            downstream[0]->T = tmp;
        }
    }
};

class CE_OP:public Operation{
public:
    CE_OP(Tensor* O, Tensor*Y):Operation(1, 1){
        op_name = "Cross-Entropy";
        downstream[0] = O;
        upstream[0] = Y;
    }
    void forward() override{
        // For CE_OP, forward is SoftMax
        softMax(downstream[0], downstream[0]);
    }
    void backward() override{
        if(!downstream[0]->dT){
            addGrad(downstream[0]);
        }        
        
        // Let's create a temporary Tensor pointing to downstream[0]->dT
        Tensor* d_der_O = makeTensorbyShape(downstream[0], downstream[0]->device_type);
        d_der_O->T = downstream[0]->dT; // Trick: point T to dT
        
        CESoftmax_deriv(d_der_O, downstream[0], upstream[0]);
        
        // Don't free d_der_O->T as it points to downstream[0]->dT
        d_der_O->T = NULL; 
        freeTensor(d_der_O);
    }
};
