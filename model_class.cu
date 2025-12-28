#include "operations.cu"

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
    void update(float learning_rate){
        int updated_count = 0;
        for(int i = 0; i < data_len; i++){
            // Only update parameters (weights/biases)
            if(datas[i] && datas[i]->isParam){
                updateTensor(datas[i], learning_rate); 
                updated_count++;
            }
        }
    }
    void printModel(){
        for(int i=0; i < op_len; i++){
            cout <<i<<" : "<< operations[i]->op_name << endl;
        }
    }
    void zero_grad(){
        for(int i = 0; i < data_len; i++){
            if(datas[i] && datas[i]->dT){
                if(datas[i]->device_type){
                    cudaSetDevice(datas[i]->device_type - 1);
                    cudaMemset(datas[i]->dT, 0, datas[i]->sizeTensor * sizeof(float));
                }else{
                    memset(datas[i]->dT, 0, datas[i]->sizeTensor * sizeof(float));
                }
            }
        }
    }
};
