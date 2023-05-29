CIFAR10_CUDA: main.cu cifar10_reader.cu matrix.cu
	nvcc -std=c++11 -o test main.cu cifar10_reader.cu matrix.cu
