#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 8

typedef struct Tensor{
    float *T;           //명심해라. 이건 배열이 아니라 시작주소이다.
    int *dim;           //dim. stride와는 다르다. subTensor에서는 완전 다르다. 
    int *stride;        //다음 차원 얼마나 건너뛰어야하는지 알려줌.
    int *d_dim_stride;  //GPU에 정보제공. 만약 CPU에 있으면 NULL.
    int num_dim;        //dim, stride, d_dim_s~ 의 길이 정보제공
    int sizeTensor;     //텐서 크기
    char device_type;   //텐서위치
    char isSub;         //subTensor인지 확인해줌.
}Tensor;

//만들고
Tensor *makeTensor(int *dim, int num_dim, int device_type);
//모양따라 만들고.
Tensor *makeTensorbyShape(Tensor* src, int device_type);

//약간 커서같은 느낌. 그 부분의 값을 공유.
Tensor *makeSubTensor(Tensor* src, int* start_point, int* dim, int num_dim);

//메모리 해제
void freeTensor(Tensor *ten);

//reshape하면서 값도 복사해줌. 아마도 느림. src는 subTensor이여도 된다.
//부분 복사를 할 때,
//"copyReshapeTensor(makeTensor(subDim,num_dim, 1), makeSubTensor(src, sp, subDim, num_dim));"이런식으로 복사를 해줄 수 있다.
Tensor* copyReshapeTensor(Tensor* dst, Tensor* src, int* reshape);

//얘는 dst, src모두 완전한Tensor 이어야한다. 대신 빠름.
Tensor* copyTensor(Tensor* dst, Tensor* src);

//값 출력. CPU만 된다.
void printTensor(Tensor *ten);
//정보 출력. 다 된다.
void infoTensor(Tensor *ten);


Tensor* compMatMul(Tensor*dC, Tensor* dA, Tensor* dB);

Tensor* matmul(Tensor* dC, Tensor *dA, Tensor* dB);
Tensor* matmul_matwise(Tensor* dC, Tensor *dA, Tensor* dB);
#endif // TENSOR_H

//주의할 점. 
//어차피 할당된 곳을 계속 쓰게 되어있다. 굳이 free할 일이 거의 없으므로 웬만하면 inline 또는 dst, src 꼴로 만들어 주는 것이 제일 좋다.
