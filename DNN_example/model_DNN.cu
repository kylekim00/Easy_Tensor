#include<iostream>
#include<stdlib.h>
#include "./../easy_tensor.h"
#include<string.h>

using namespace std;

#include "../model_class.cu"

#include <time.h>

int main(){

    //GPU memory
    int batch_size = 32;
    int input_dim = 784; // 28x28
    int layer_dim[5] = {784, 50, 30, 40, 10};

    int in_dim[2];

    // Host Tensors
    in_dim[0] = batch_size;
    in_dim[1] = input_dim;
    Tensor* input = mallocTensor(in_dim, 2, 0); // CPU

    in_dim[0] = batch_size;
    Tensor* label = mallocTensor(in_dim, 1, 0); // CPU
    
    // Device Tensors
    in_dim[0] = batch_size;
    in_dim[1] = input_dim;
    Tensor* d_input = mallocTensor(in_dim, 2, 1); // GPU
    
    in_dim[0] = batch_size;
    Tensor* d_label = mallocTensor(in_dim, 1, 1); // GPU

    Tensor* d_W[4];
    Tensor* d_b[4];
    Tensor* d_A[5]; // Activations
    d_A[0] = d_input;

    Model m1;
    
    srand(time(NULL));

    for(int i=0; i < 4; i++){
        // Weight: [in_dim, out_dim] -> [784, 50]
        in_dim[0] = layer_dim[i];
        in_dim[1] = layer_dim[i+1];
        d_W[i] = mallocTensor(in_dim, 2, 1);
        d_W[i]->isParam = 1;

        // Init Weight (Load from file to match nn.cu)
        char file_name[50];
        file_name[0] = 2*i + '0';
        strcpy(file_name+1, "_init_blocks.bin");
        // We need a host tensor to load into first? copyTensorfromFILE reads into dst->T. 
        // dst must be CPU tensor.
        Tensor* w_temp = mallocTensor(in_dim, 2, 0); 
        copyTensorfromFILE(w_temp, file_name);
        copyTensor(d_W[i], w_temp); // Copy to Device
        freeTensor(w_temp);

        // Bias: [out_dim]
        in_dim[0] = layer_dim[i+1];
        d_b[i] = mallocTensor(in_dim, 1, 1);
        d_b[i]->isParam = 1;
        
        // Init Bias (Load from file)
        file_name[0] = 2*i+1 + '0';
        strcpy(file_name+1, "_init_blocks.bin");
        Tensor* b_temp = mallocTensor(in_dim, 1, 0);
        copyTensorfromFILE(b_temp, file_name);
        copyTensor(d_b[i], b_temp);
        freeTensor(b_temp);

        // Activation for next layer: [batch_size, out_dim]
        in_dim[0] = batch_size;
        in_dim[1] = layer_dim[i+1];
        d_A[i+1] = mallocTensor(in_dim, 2, 1);
        
        m1.addOperation(new Matmul_bias_OP(d_A[i+1], d_A[i], d_W[i], d_b[i]));
        
        if(i < 3){
             m1.addOperation(new ReLU_OP(d_A[i+1]));
        }
    }
    
    // Output tensor (last activation)
    m1.addOperation(new CE_OP(d_A[4], d_label));

    //==============================TRAIN===========================================
    // For calculating accuracy on CPU, we need O on CPU
    in_dim[0] = batch_size;
    in_dim[1] = layer_dim[4];
    Tensor* O = mallocTensor(in_dim, 2, 0);

    FILE * data_file, *label_file;
    float learning_rate = 0.0001;
    
    printf("Start Training\n");

    for(int iter=0; iter < 30; iter++){//iteration

        data_file = LoaderINIT("data_norm.bin");
        label_file = LoaderINIT("label.bin");

        double loss = 0;
        int accuracy = 0; // Total correct predictions
        
        int num_batches = 60000/batch_size;
        
        for(int batch=0; batch < num_batches; batch++){//batch
            m1.zero_grad();
            // Load Data
            copyTensor(d_input, LoaderNEXT(input, data_file));
            copyTensor(d_label, LoaderNEXT(label, label_file));

            m1.forward();
            
            // Calculate Loss/Accuracy
            copyTensor(O, d_A[4]);
            
            loss += CrossEntropyLoss(O, label);
            
            // Calculate Accuracy
            accuracy += accuracy_CPU(O, label);
            
            float current_acc = (float)accuracy / ((batch + 1) * batch_size) * 100;

            m1.backward();
            
            if(batch < 50000/batch_size){ // Training phase
                m1.update(learning_rate);
            }
            
            print_progress(batch, num_batches, current_acc);
        }
        printf("\nIteration: %d, Loss: %.4f, Accuracy: %.2f%%\n", iter, loss/num_batches, (float)accuracy/60000 * 100);
        LoaderCLOSE(data_file);
        LoaderCLOSE(label_file);
    }

    return 0;
}