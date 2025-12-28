
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "./../easy_tensor.h"

// Helper functions now in easy_tensor_struct.cu


int main(){
    int batch_size = 32;
    float learning_rate = 0.0001;
    int layer_dim[] = {784, 50, 30, 40, 10};
    int in_dim[] = {batch_size, layer_dim[0]};

    //==========input allocation========================
    Tensor* input = mallocTensor(in_dim, 2, 0);
    Tensor* d_input = makeTensorbyShape(input, 1);
    in_dim[0] = 784;
    in_dim[1] = batch_size;
    Tensor* d_input_t = mallocTensor(in_dim, 2, 1);

    Tensor* label = mallocTensor(&batch_size, 1, 0);
    Tensor* d_label = makeTensorbyShape(label, 1);
    //==================================================

    Tensor* W[4];
    Tensor* d_W[4];
    Tensor* b[4];
    Tensor* d_b[4];

    Tensor* d_der_W[4];
    Tensor* d_W_t[4];
    Tensor* d_der_b[4];

    Tensor* d_A[4];
    Tensor* d_der_A[4];
    Tensor* d_A_t[4];
    //이건 0-9사이에 가중치만 있으므로 가능
    //==========================weight initialization===============================
    char file_name[] = "0_init_blocks.bin";
    for(int i=0; i < sizeof(W)/sizeof(Tensor*); i++){

        file_name[0] = 2*i + '0';
        W[i] = copyTensorfromFILE(mallocTensor(layer_dim+i, 2, 0), file_name);
        d_W[i] = copyTensor(makeTensorbyShape(W[i], 1), W[i]);
        d_der_W[i] = makeTensorbyShape(W[i], 1);

        in_dim[0] = layer_dim[i+1];
        in_dim[1] = layer_dim[i];
        d_W_t[i] = mallocTensor(in_dim, 2, 1);

        file_name[0] = 2*i+1 + '0';
        b[i] = copyTensorfromFILE(mallocTensor(layer_dim+i+1, 1, 0), file_name);
        d_b[i] = copyTensor(makeTensorbyShape(b[i],1),b[i]);
        d_der_b[i] = makeTensorbyShape(b[i], 1);
        
        in_dim[0] = batch_size;
        in_dim[1] = layer_dim[i+1];
        d_A[i] = mallocTensor(in_dim, 2, 1);
        d_der_A[i] = makeTensorbyShape(d_A[i], 1);

        in_dim[0] = layer_dim[i+1];
        in_dim[1] = batch_size;
        d_A_t[i] = mallocTensor(in_dim, 2, 1);
        // infoTensor(d_A_t[i]);
    }
    
    in_dim[0] = batch_size;
    in_dim[1] = layer_dim[sizeof(layer_dim)/sizeof(int)-1];
    Tensor* O = mallocTensor(in_dim, 2, 0);
    Tensor* d_O = makeTensorbyShape(O, 1);
    //===============================================================================


    //=============dataLoader========================
    FILE * data_file, *label_file;
    //===============================================


    //==============================TRAIN===========================================
    for(int iter=0; iter < 30; iter++){//iteration

        data_file = LoaderINIT("data_norm.bin");
        label_file = LoaderINIT("label.bin");

        double loss = 0;
        int accuracy = 0;
        for(int batch=0; batch < 60000/batch_size;batch++){//batch
            d_input = copyTensor(d_input, LoaderNEXT(input, data_file));
            d_label = copyTensor(d_label, LoaderNEXT(label, label_file));//both label and d_label is written here.

            //forward pass
            d_A[0] = matmul_bias(d_A[0], d_input, d_W[0], d_b[0], 0);

            for(int i=1; i < sizeof(d_W)/sizeof(Tensor*);i++){
                ReLU_inline(d_A[i-1]);
                d_A[i] = matmul_bias(d_A[i], d_A[i-1], d_W[i], d_b[i], 0);
            }

            //loss
            d_O = softMax(d_O, d_A[sizeof(d_A)/sizeof(Tensor*)-1]);
            loss += CrossEntropyLoss(copyTensor(O, d_O), label);//cross entropy loss

            //backward pass
            if(batch < 50000/batch_size){
                d_der_A[sizeof(d_der_A)/sizeof(Tensor*) - 1] = CESoftmax_deriv(d_der_A[sizeof(d_der_A)/sizeof(Tensor*) - 1], d_O, d_label);

                for(int i = sizeof(d_der_W)/sizeof(Tensor*) - 1; i >=1; i--){   //3, 2, 1,
                    d_der_W[i] = matmul(d_der_W[i], copyTransposeTensor(d_A_t[i-1],d_A[i-1]), d_der_A[i]);
                    d_der_b[i] = rowcolwise_sum(d_der_b[i], d_der_A[i], 0);
                    
                    d_der_A[i-1] = matmul(d_der_A[i-1], d_der_A[i], copyTransposeTensor(d_W_t[i], d_W[i]));
                    d_der_A[i-1] = elementWise_Tensor(d_der_A[i-1] ,d_der_A[i-1],'m', d_A[i-1]);//dReLU가 들어가야함. d_der_A[i] = d_A[i]==0 ? 0 : d_der_A[i];
                }
                
                //update
                d_der_W[0] = matmul(d_der_W[0], copyTransposeTensor(d_input_t,d_input), d_der_A[0]);
                d_der_b[0] = rowcolwise_sum(d_der_b[0], d_A[0], 0);

                for(int i=0; i < sizeof(d_der_W)/sizeof(Tensor*); i++){
                    d_W[i] = elementWise_Tensor(d_W[i],d_W[i],'-',scalar_Tensor(d_der_W[i], '*',learning_rate));
                    d_b[i] = elementWise_Tensor(d_b[i],d_b[i],'-',scalar_Tensor(d_der_b[i], '*',learning_rate));
                }
            }else{//iter
                accuracy += accuracy_CPU(O, label);
            }
            
            print_progress(batch, 60000/batch_size, 0.0f);

        }
        
        printf("\nvalid acc: %d/10000 | %.2f%%\nloss: %f\n",accuracy, (float)accuracy/100,loss/(60000/batch_size));

        LoaderCLOSE(data_file);
        LoaderCLOSE(label_file);
        printf("\n");
    }

    //free Weights///////////////////////////////////////////////////////////////////////////////////////
    for(int i=0; i <sizeof(W)/sizeof(Tensor*);i++){
        freeTensor(W[i]);
        freeTensor(d_W[i]);
        freeTensor(d_der_W[i]);
        freeTensor(d_W_t[i]);

        freeTensor(b[i]);
        freeTensor(d_b[i]);
        freeTensor(d_der_b[i]);

        freeTensor(d_A[i]);
        freeTensor(d_der_A[i]);
        freeTensor(d_A_t[i]);
    }
    
    

    freeTensor(d_input);
    freeTensor(input);
    

    
    
    
}