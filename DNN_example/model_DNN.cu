#include<iostream>
#include<stdlib.h>
#include "./../easy_tensor.h"
#include<string.h>

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
                elementWise_Tensor(downstream[0], downstream[0], '+', down_tmp0);
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
                
                elementWise_Tensor(downstream[0], downstream[0], '+', down_tmp0);
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
    Matmul_bias_OP(Tensor* Y, Tensor* X1, Tensor* X2, Tensor* bias) : Operation(1, 3){
        op_name = "matmul_bias";
        upstream[0] = Y;
        downstream[0] = X1;
        downstream[1] = X2;
        downstream[2] = bias;
    }

    void forward() override{
        matmul_bias(upstream[0], downstream[0], downstream[1], downstream[2], 0);
    }

    void backward() override{
        if(upstream[0]->dT){
            //downstream[0] deriv
            ///Transpose downstream[0] tmp
            Tensor* down0_T = makeTensorbyTransposedShape(downstream[0], downstream[1]->device_type);
            copyTransposeTensor(down0_T, downstream[0]);

            //transpose downstream[1] tmp
            Tensor* down1_T = makeTensorbyTransposedShape(downstream[1], downstream[1]->device_type);
            copyTransposeTensor(down1_T, downstream[1]);


            //derivative가 없을 경우 grad 저장공간 할당, 있을 경우 기존 grad를 더해준다. 
            Tensor* down_tmp0 = NULL, *down_tmp1 = NULL, *down_tmp2 = NULL;
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
            if(!downstream[2]->dT){//만약 x1이 deriv가 없다.
                addGrad(downstream[2]);
            }else{
                down_tmp2 = makeTensorbyShape(downstream[2], downstream[2]->device_type);
                down_tmp2 = copyTensor_grad(down_tmp2, downstream[2]);
            }

            //@X1 = @Y x X2^T
            matmul_grad(downstream[0], 1,
                        upstream[0], 1,
                        down1_T, 0
                        );

            //@X2 = X1^T x @Y
            matmul_grad(downstream[1], 1,
                        down0_T, 0,
                        upstream[0], 1);
            //bias
            // rowcolwise_sum();
            rowcolwise_sum_grad(
                        downstream[2], GRAD_TRUE,
                        upstream[0], GRAD_TRUE,
                        0
            );
            if(down_tmp0){
                
                elementWise_Tensor(downstream[0], downstream[0], '+', down_tmp0);
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

            freeTensor(down0_T);
            freeTensor(down1_T);
        }
        // else{
        //     Tensor*up = makeTensorbyShape(upstream[0], upstream[0]->device_type);
        //     // 1로 채운 upstream 크기의 Tensor
        //     if(up->device_type)
        //         reset_Tensor(up, 1);
        //     else{
        //         for(int i=0; i < up->sizeTensor; i++)
        //             up->T[i] = 1;
        //     }
            

        //     //////////////Transpose downstream////////////////
            
        //     ///Transpose downstream[0] tmp
        //     Tensor* down0_T = makeTensorbyTransposedShape(downstream[0], downstream[1]->device_type);
        //     copyTransposeTensor(down0_T, downstream[0]);

        //     //transpose downstream[1] tmp
        //     Tensor* down1_T = makeTensorbyTransposedShape(downstream[1], downstream[1]->device_type);
        //     copyTransposeTensor(down1_T, downstream[1]);

        //     // printTensor(copyTensor(makeTensorbyShape(down0_T, 0), down0_T));
        //     //derivative가 없다.
            
        //     //derivative가 없을 경우 grad 저장공간 할당, 있을 경우 기존 grad를 더해준다. 
        //     Tensor* down_tmp0 = NULL, *down_tmp1 = NULL, *down_tmp2 = NULL;
        //     if(!downstream[0]->dT){//만약 x2가 deriv가 없다. 
        //         addGrad(downstream[0]);
        //     }else{
        //         down_tmp0 = makeTensorbyShape(downstream[0], downstream[0]->device_type);
        //         down_tmp0 = copyTensor_grad(down_tmp0, downstream[0]);
        //         // printTensor_grad(copyTensor_grad(makeTensorbyShape(downstream[0], 0), downstream[0]));
        //         // printTensor_grad(copyTensor_grad(makeTensorbyShape(down_tmp0, 0), down_tmp0));
        //     }
        //     if(!downstream[1]->dT){//만약 x1이 deriv가 없다.
        //         addGrad(downstream[1]);
        //     }else{
        //         down_tmp1 = makeTensorbyShape(downstream[1], downstream[1]->device_type);
        //         down_tmp1 = copyTensor_grad(down_tmp1, downstream[1]);
        //     }
        //     if(!downstream[2]->dT){//만약 x1이 deriv가 없다.
        //         addGrad(downstream[2]);
        //     }else{
        //         down_tmp2 = makeTensorbyShape(downstream[2], downstream[2]->device_type);
        //         down_tmp2 = copyTensor_grad(down_tmp1, downstream[2]);
        //     }

        //     //@X2 = X1^T x @Y
        //     matmul_grad(downstream[1], 1,
        //                 down0_T, 0,
        //                 up, 0);
        //     //@X2 = X1^T x @Y
        //     matmul_grad(downstream[0], 1,
        //                 up, 0,
        //                 down1_T, 0
        //                 );
            
        //     if(down_tmp0){
        //         elementWise_Tensor(downstream[0], downstream[0], '+', down_tmp0);
        //         downstream[1] = elementWise_Tensor_grad(downstream[0], GRAD_TRUE, downstream[0], GRAD_TRUE,'+' , down_tmp0, GRAD_TRUE);
        //         freeTensor(down_tmp0);
        //     }
        //     if(down_tmp1){
        //         downstream[1] = elementWise_Tensor_grad(downstream[1], GRAD_TRUE, downstream[1], GRAD_TRUE,'+' , down_tmp1, GRAD_TRUE);
        //         freeTensor(down_tmp1);
        //     }

        //     freeTensor(down0_T);
        //     freeTensor(down1_T);

        //     freeTensor(up);
        // }
    }
};





//////////////////////////////////model/////////////////////////////////

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
        softMax(downstream[0], downstream[0]);
        // cout << "loss" << endl;
    }
    void backward() override{
        if(!downstream[0]->dT){
            addGrad(downstream[0]);
        }        
        elementWise_Tensor_grad_2(
            downstream[0], GRAD_TRUE,
            downstream[0], GRAD_FALSE,
            '-',
            upstream[0], GRAD_FALSE
        );
    }
};

class Model{
public:
    Operation** operations;
    int op_len;
    Tensor** datas;
    int data_len;
    Model(){
        operations = nullptr;
        op_len = 0;
        datas = 0;
        datas = nullptr;
        data_len = 0;
    }

    void addOperation(Operation* op){
        if(op == nullptr){
            cout << "Model : op Not Appropriate" << endl;
        }
        if(op_len % 10 == 0){
            Operation** op_tmp = operations;
            operations = (Operation**)malloc(sizeof(Operation*) * (op_len + 10));
            for(int i=0; i < op_len; i++){
                operations[i] = op_tmp[i];
            }
            free(op_tmp);
        }
        operations[op_len] = op;
        op_len++;

        for(int i=0; i < op->upstream_len; i++){
            char flag = 1;
            for(int j=0; j < data_len; j++){
                if(datas[j] == op->upstream[i]){
                    flag = 0;
                    break;
                }
            }
            if(flag){
                if(data_len % 10 ==0){
                    Tensor** tmp = datas;
                    datas = (Tensor**)malloc(sizeof(Tensor*) * (data_len + 10));
                    for(int k=0; k < data_len; k++){
                        datas[k] = tmp[k];
                    }
                    free(tmp);
                }
                datas[data_len] = op->upstream[i];
                data_len++;
            }
        }
        for(int i=0; i<op->downstream_len; i++){
            char flag = 1;
            for(int j=0; j < data_len; j++){
                if(datas[j] == op->downstream[i]){
                    flag = 0;
                    break;
                }
            }
            if(flag){
                if(data_len % 10 ==0){
                    Tensor** tmp = datas;
                    datas = (Tensor**)malloc(sizeof(Tensor*) * (data_len + 10));
                    for(int k=0; k < data_len; k++){
                        datas[k] = tmp[k];
                    }
                    free(tmp);
                }
                datas[data_len] = op->downstream[i];
                data_len++;
            }
        }
    }

    void forward(){
        if(operations != nullptr){
            for(int i=0; i < op_len; i++){
                operations[i]->forward();
            }
        }
    }
    void backward(){
        if(operations != nullptr){
            for(int i = op_len-1; i >= 0; i--){
                // cout << operations[i]->getOpname() << endl;
                operations[i]->backward();
            }
        }
    }
    void update(){
        for(int i = 0; i < data_len; i++){
            if(datas[i])
                updateTensor(datas[i], 0.001);
        }
    }
    void printModel(){
        for(int i=0; i < op_len; i++){
            cout <<i<<" : "<< operations[i]->op_name << endl;
        }
    }
};



Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = 0.0001 * i - 25/2;
    }
    return ten;
}
Tensor* dummyTensor2(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = (i % ten->stride[0] == 1) ? 1 : 0;
    }
    return ten;
}
int main(){
    Tensor* A = dummyTensor(makeTensor("728 728", 0));
    Tensor* B = dummyTensor(makeTensor("728 30", 0));
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    Tensor* bias = dummyTensor(makeTensor("30", 0));
    Tensor* dBias = copyTensor(makeTensorbyShape(bias, 1), bias);
    Tensor* C = makeTensor("728 30", 0);
    Tensor* D = makeTensor("728 30", 0);
    Tensor* Y = dummyTensor2(makeTensor("728 30", 0));
    Tensor* dY = copyTensor(makeTensorbyShape(Y, 1), Y);
    Tensor* dC = makeTensorbyShape(C, 1);
    Tensor* dD = makeTensorbyShape(D, 1);

    Model m1;
    m1.addOperation(new Matmul_bias_OP(dC, dA, dB, dBias));
    m1.addOperation(new ReLU_OP(dC));
    m1.addOperation(new Matmul_OP(dD, dA, dC));
    m1.addOperation(new CE_OP(dD, dY));
    
    for(int i=0; i < 2; i++){
        m1.forward();
        m1.backward();
        m1.update();
    }


    
    // printTensor(copyTensor_grad(D, dD));
    // printTensor_grad(D);
    
    // m1.update();
    

    
    // cout << dC << dA << dB << endl;
    // m1.operations[0]->forward();
    // // m1.operations[1]->forward();
    // m1.operations[2]->forward();

    // m1.operations[2]->backward();
    // // m1.operations[1]->backward();
    // m1.operations[0]->backward();

    // printTensor(copyTensor(makeTensorbyShape(dC, 0), elementWise_Tensor(dC, dB,'m' ,dA)));
    // printTensor_grad(copyTensor_grad(A, dA));
    // printTensor_grad(copyTensor_grad(B, dB));
    // printTensor_grad(copyTensor_grad(bias, dBias));
    // printTensor_grad(copyTensor_grad(C, dC));
    // cout << "asdf" << endl;
    // printTensor(copyTensor(C, dC));
    
}