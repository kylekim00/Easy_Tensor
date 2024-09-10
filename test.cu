#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = i;
        ten->T[i] = dm;
    }
    return ten;
}

int main(){
    int dim[] = {5, 4, 2, 3};
    Tensor* A = makeTensor(dim, 4, 0);
    A = dummyTensor(A);
    int trans_dim[] = {2, 3, 4, 5};
    Tensor* At = makeTensor(trans_dim, 4, 0);
    
    Tensor*dA = copyTensor(makeTensorbyShape(A, 1), A);

    Tensor*dAt = makeTensorbyShape(At, 1);

    int shape[] = {2, 3, 1, 0};

    dAt = copyReshapeTensor(dAt, dA, shape);
    At = copyTensor(At, dAt);

    int sp[] = {0, 1, 1, 1};
    int sub_dim[] = {2, 2, 2, 3};
    Tensor* subdAt = makeSubTensor(dAt, sp, sub_dim, 4);
    for(int i=0; i < 4; i++){
        shape[i] = i;
    }
    printTensor(copyTensor(makeTensorbyShape(subdAt, 0),copyReshapeTensor(makeTensorbyShape(subdAt, 1), subdAt, shape)));
    printTensor(At);


    

    // Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    // Tensor* dAt = makeTensorbyShape(At, 1);

    // copyReshapeTensor(dAt, dA, resh);
    // copyTensor(At, dAt);
    // printTensor(At);

    // int sp[] = {0, 0, 2};//batch, numToken, Q
    // int sub_Dim[] = { 2, 2};
    // Tensor* subA = makeSubTensor(A, sp, sub_Dim, 2);
    // printTensor(subA);
    // freeTensor(subA);
    // sp[2] = 4;
    
    // subA = makelightcopysubTensor(A, sp, sub_Dim, 2);
    // printTensor(subA);
    // freeTensor(subA);

    freeTensor(A);
}