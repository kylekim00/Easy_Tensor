#include<stdio.h>
#include<stdlib.h>
#include"./../easy_tensor.h"
#include<cuda_runtime.h>

///////////////////////////DATALOADER///////////////////////////////////

FILE* LoaderINIT(const char* file_name){
    char f_name[50] = "./data/";
    int len = strlen(f_name);
    int i;
    for(i=0; file_name[i]; i++){
        f_name[i+len] = file_name[i];
    }
    f_name[i+len] = '\0';
    
    FILE *file = fopen(f_name, "rb");
    if (!file) {
        printf("Error opening file\n");
        return NULL;
    }
    return file;
}


Tensor* LoaderNEXT(Tensor* dst, FILE*file){
    if(dst->device_type){
        printf("Tensor must be on CPU\n");
        return NULL;
    }
    size_t num_elements = fread(dst->T, sizeof(float), dst->sizeTensor, file);
    if (num_elements != dst->sizeTensor) {
        printf("Error reading file\n");
        return NULL;
    }
    return dst;
}

void LoaderCLOSE(FILE* file){
    fclose(file);
}

int main(){
    FILE* data_file = LoaderINIT("data_norm.bin");
    FILE* label_file = LoaderINIT("label.bin");
    Tensor* label = makeTensor("16", 0);
    Tensor* data = makeTensor("16 784", 0);

    Tensor* d_label = makeTensorbyShape(label, 1);
    Tensor* d_data = makeTensorbyShape(data, 1);
    // for(int i=0; i < 2; i++)
    for(int i=0; i< 60000/label->sizeTensor; i++){
        d_data = copyTensor(d_data,LoaderNEXT(data, data_file));
        d_label = copyTensor(d_label, LoaderNEXT(label, label_file));
        /////////////////model///////////////////





        /////////////////////////////////////////
    }


    freeTensor(printTensor(makeSubTensor(data, "0 128", "16 8")));
    printTensor(label);
    LoaderCLOSE(data_file);
    LoaderCLOSE(label_file);
    freeTensor(data);
    freeTensor(label);
    freeTensor(d_data);
    freeTensor(d_label);
}