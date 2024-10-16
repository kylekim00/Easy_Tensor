#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"

#define FORWARD 0
#define BACKWARD 1

#define MODEL_OPSNUM_RE 30


//Operation은 Tensor가 연산을 하는 그 과정 자체를 struct로 만들어 저장을 한다.
typedef struct Operation{
    int op;
    int upstream_num;
    int downstream_num;
    Tensor* upstream[5];
    Tensor* downstream[5];
}Operation;

//Model은 사실상 Opration의 배열 역할을 한다고 보면 된다. 물론 추가적인 정보를 나중에 줄 수도 있지만. 결국 미분 역전파는 순차적으로 이루어지기 때문에 for문을 반대로 돌리면 끝나므로 걍 이렇게 만들면 됌
typedef struct Model{
    int ops_num;
    struct Operation** root;
}Model;

//모델 struct를 만드는 것
Model* makeModel(){
    Model* newModel = (Model*)malloc(sizeof(Model));
    newModel->ops_num = 0;
    newModel->root = NULL;
    return newModel;
}


//이거는 operation을 append해줄 때 만약 model에 operation의 수가 부족 할 경우 배열을 늘려주는 역할을 한다.
Model* appendOperation(Model* model, Operation* op){
    model->ops_num++;
    if(model->ops_num % MODEL_OPSNUM_RE == 0){
        Operation** tmp = model->root;
        model->root = (Operation**)malloc(sizeof(Operation*) * MODEL_OPSNUM_RE);
        for(int i=0; i < model->ops_num; i++){

        }
    }
}

//이거는 dT가 null인 Tensor한테 determinant 공간을 추가해주는 역할을 한다. 다만 subTensor일 때를 생각은 해봐야 할듯. 같은 공가늘근데 같은 공간을 사용하는데 그럼 그냥 원래 Tensor의 공간에 그냥 미분 값을 넣어도 되지 않을까....?????

Tensor* addDeterminant(Tensor* ten){
    ten->dT = (float*)malloc(sizeof(float) * ten->sizeTensor);
}




Tensor* Op_matmul(Tensor*dC, Tensor*dA, Tensor*dB, Model*model, char propagation_STATE){
    if(model == NULL){
        printf("model is NULL\n");
        return NULL;
    }
    Operation* newOp = (Operation*)malloc(sizeof(Operation));
    if(propagation_STATE == FORWARD){
        matmul(dC, dA, dB);
    }else if(propagation_STATE == BACKWARD){
        if(dC->dT != NULL){
            
        }
    }


    if(newOp)
        appendOperation(model, newOp);
}
 
int main(){
    Model* model = makeModel();


}