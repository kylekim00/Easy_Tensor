#include<stdlib.h>
#include<iostream>
#include"easy_tensor.h"

#define FORWARD 0
#define BACKWARD 1

#define MODEL_OPSNUM_RE 30

#define OPERATION_MATMUL 0

#define OPERATION_UP_IDX_A 0
#define OPERATION_UP_IDX_B 1

#define OPERATION_DOWN_IDX_C 0

typedef enum OPCODE{
        MATMUL,
        MATMUL_BIAS,
        RELU_INLINE,
        ROWCOLWISE_SUM,
        SCALAR_TENSOR,
        LINEAR,
        RELU
}OPCODE;

//Operation은 Tensor가 연산을 하는 그 과정 자체를 struct로 만들어 저장을 한다.
// typedef struct Operation{
//     int op;
//     int upstream_num;
//     int downstream_num;
//     Tensor** upstream;
//     Tensor** downstream;
// }Operation;

class Operation{
    protected:
    int opcode;
    int upstream_num;
    int downstream_num;
    Tensor**upstream;
    Tensor**downstream;
    public:
    Operation(int opcode, int upstream_num, int downstream_num){
        this->opcode = opcode;
        this->upstream_num = upstream_num;
        this->downstream_num = downstream_num;
    }
    int getUpnum(){
        return upstream_num;
    }
    int getDownnum(){
        return downstream_num;
    }
    Tensor* forward();

    Tensor* backward();
};

class Linear : public Operation{
    
    Linear(Tensor* O, Tensor* W, Tensor* X, Tensor* b):Operation(LINEAR, 1, 3){
        this->upstream = (Tensor**)malloc(sizeof(Tensor*) * this->getUpnum());
        this->upstream[0] = O;
        this->downstream = (Tensor**)malloc(sizeof(Tensor*)* this->getDownnum());
        this->downstream[0] = W;
        this->downstream[1] = X;
        this->downstream[2] = b;
    }
    public :
    void forward(){
        this->upstream[0] = matmul_bias(this->upstream[0], this->downstream[0], this->downstream[1], this->downstream[2], 0);
    }
    void backward(){

    }
};

class ReLU : protected Operation{
    ReLU(Tensor* A, Tensor* Z):Operation(RELU, 1, 1){
        this->upstream = (Tensor**)malloc(sizeof(Tensor*) * this->getUpnum());
        this->downstream = (Tensor**)malloc(sizeof(Tensor*)* this->getDownnum());
        this->upstream[0] = 
        
    }
};

//Model은 사실상 Opration의 배열 역할을 한다고 보면 된다. 물론 추가적인 정보를 나중에 줄 수도 있지만. 결국 미분 역전파는 순차적으로 이루어지기 때문에 for문을 반대로 돌리면 끝나므로 걍 이렇게 만들면 됌
class Model{
    Operation** root;
    int UP_size = 10;
    int op_num = 0;
    public:
    Model(){
        this->root = (Operation**)malloc(sizeof(Operation*) * this->UP_size);
    }
    void addOperation(Operation* op){
        
    }
};
int main(){

    Model* model = new Model();
    throw std::invalid_argument("Invalid parameter: value must be greater than zero.");
    return 0;
    // Operation* op1 = makeOperation(MATMUL, 1, 1);

}