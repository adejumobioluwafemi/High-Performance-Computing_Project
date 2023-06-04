## HPC_Project_2023_Skt 

#### CUDA Accelerated Training of Multilayer Perceptron

### Authors:

Koffivi F. Gbagbe & Oluwafemi P. Adejumobi

### Execution:
0. Download the CIFAR-10 dataset for C/C++ from https://www.cs.toronto.edu/~kriz/cifar.html
1. Compile using the Makefile
2. Run download_CIFAR.py to dowload the CIFAR-10 Dataset for python
3. Run the run.sh file  to mesure execution time 

### Files structure and description

![Alt text](https://github.com/YodaX369/HPC_Project_2023_Skt/blob/main/file_stucture.png?raw=true)

> statistics_test.csv contains cuda test accuracy per epoch

> statistics_train.csv  contains cuda train accuracy per epoch

> cuda_output.txt is the output of the cuda model

> pytorch_output.txt is the output of the pytorch model

> Run_Output.png terminal sceenshot of the execution time of the two model
 (the first one is the cuda version and the second one is the pytorch version)
![Alt text](https://github.com/YodaX369/HPC_Project_2023_Skt/blob/main/Run_Output.png?raw=true)


### To Do:

Add comparison plot for test and train accuracy of Cuda and Pytorch using: statistics_test.csv, statistics_train.csv and pytorch_output.txt



