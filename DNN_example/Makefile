
CC = nvcc
CFLAGS = -I../
SRCS = model_DNN.cu ../easy_tensor_struct.cu
TARGET = model_dnn

all: $(TARGET) nn

$(TARGET): $(SRCS)
	$(CC) -o $@ $^ $(CFLAGS)

nn: nn.cu ../easy_tensor_struct.cu
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(TARGET) nn a.out
