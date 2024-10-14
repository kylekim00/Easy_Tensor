#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = i;
    }
    return ten;
}

Tensor* dummyTensor2(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = 2;
    }
    return ten;
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
    // dA = normalize(dA, dA);
    // printTensor(makeSubTensor(copyTensor(A, dA),"0,0,0", "8 8"));

    Tensor *A = dummyTensor(makeTensor("4 3 5 5", 0));
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    
    Tensor* B = dummyTensor(makeTensor("5 5",0));
    Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);

    // Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* C = makeTensorbyShape(A, 0);
    Tensor* dC = copyTensor(makeTensorbyShape(C, 1), C);

    printTensor(A);
    // printTensor(elementWise_Tensor(C, A, '+', B));
    printTensor(copyTensor(C, elementWise_Tensor(dC, dB,'*',dA)));



    Tensor* (*dd)(Tensor*);
    dd = printTensor;
    dd(A);
}