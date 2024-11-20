#include<iostream>
using namespace std;

class Data{
public:
    int data;
    int deriv;

    Data(int data=0){
        this->data = data;
        deriv = 1;

    }
    int getData(){
        return data;
    }
};

class Operation{
public:
    string op_name;
    Data** upstream;
    Data** downstream;
    int upstream_len;
    int downstream_len;

    Operation(int upstream_len, int downstream_len){
        this->op_name = "operation";
        this->upstream = (Data**)malloc(sizeof(Data*) * upstream_len);
        this->downstream = (Data**)malloc(sizeof(Data*) * downstream_len);
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

class Add_Op:public Operation{
public:
    Add_Op(Data*y, Data* x1, Data *x2) : Operation(1, 2){
        op_name = "add_op";
        upstream[0] = y;
        downstream[0] = x1;
        downstream[1] = x2;
    }

    void forward() override{
        upstream[0]->data = downstream[0]->getData() + downstream[1]->getData();
    }
    void backward(){
        downstream[0]->deriv = 1 * upstream[0]->deriv;
        downstream[1]->deriv = 1 * upstream[0]->deriv;
    }

    string getOpname(){
        return op_name;
    }
};

class Mult_Op:public Operation{
public:
    Mult_Op(Data*y, Data* x1, Data *x2) : Operation(1, 2){
        op_name = "mult_op";
        upstream[0] = y;
        downstream[0] = x1;
        downstream[1] = x2;
    }

    void forward() override{
        upstream[0]->data = downstream[0]->getData() * downstream[1]->getData();
    }
    void backward(){
        downstream[0]->deriv = downstream[1]->data * upstream[0]->deriv;
        downstream[1]->deriv = downstream[0]->data * upstream[0]->deriv;
    }

    string getOpname(){
        return op_name;
    }
    ~Mult_Op(){
        cout <<"되는게 중요한 것" <<endl;
    }
    
};

class Model{
public:
    Operation** operations;
    int op_len;
    Data** datas;
    int data_len;
    Model(){
        operations = nullptr;
        op_len = 0;
        datas = nullptr;
        data_len = 0;
    }

    void addOperation(Operation* op){
        if(op == nullptr){
            cout << "this is not appropriate\n" << endl;
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

        //Data list append
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
                    Data** tmp = datas;
                    datas = (Data**)malloc(sizeof(Data*) * (data_len + 10));
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
                    Data** tmp = datas;
                    datas = (Data**)malloc(sizeof(Data*) * (data_len + 10));
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
                cout << operations[i]->getOpname() << endl;
                operations[i]->backward();
            }
        }
    }
    void modelStructure(){
        for(int i=0; i < op_len; i++){
            cout <<i<<" : "<< operations[i]->op_name << endl;
        }
    }
    ~Model(){
        if(operations != nullptr){
            for(int i=0; i < op_len; i++){
                operations[i]->~Operation();
            }
        }
    }
};
int main(){
    Data *x = new Data(34);
    Data *a = new Data(3);
    Data *y = new Data();
    Model m1 = Model();

    m1.addOperation(new Add_Op(x, x, x));
    m1.addOperation(new Mult_Op(y, a, x));

    m1.forward();
    
    m1.backward();

    cout << "원소 총 개수 : " << m1.data_len << endl;

    m1.modelStructure();
}