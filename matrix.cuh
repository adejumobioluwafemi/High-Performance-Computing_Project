
// Header guard to prevent multiple includes
#ifndef NEW_MATRIX_CUH
#define NEW_MATRIX_CUH

// Standard header includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>

// CUDA header includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// Struct representing a matrix
struct Matrix {
    int width, height;  // Dimensions of the matrix
    double *elements;  // Pointer to the matrix elements
    
    // Default constructor
    Matrix();
    
    // Constructor with dimensions specified
    Matrix(int rows, int cols);
    
    // Method for shuffling the rows of the matrix
    void shuffle(vector<int> ridx);
};

// Function for initializing a matrix with a specified value
void Initilization(Matrix *Matrix_A, double s);

// Function for copying data from one matrix to another
void Copy_Data(Matrix *Matrix_A, Matrix Matrix_B, int start, int end, bool expand = false);

// Activation function - sigmoid
double Sigmoid_Activation(double x);

// Device function to get an element from the matrix
__device__ double Get_Mat_element(Matrix *Matrix_A, int row, int col);

// Device function to set an element in the matrix
__device__ void Set_Mat_element(Matrix *Matrix_A, int row, int col, double value);

// Kernel function for matrix multiplication
__global__ void Dot_Product_Kernel(Matrix *Matrix_A, Matrix *Matrix_B, Matrix *C, bool trans1 = false, bool trans2 = false);

// Kernel function for matrix multiplication with scalar
__global__ void Multiplication_Kernel(Matrix *Matrix_A, double b);


// Kernel function for matrix multiplication of matrices
__global__ void Multiplication_Kernel(Matrix *Matrix_A, Matrix *Matrix_B, Matrix *C);



// Kernel function for element-wise matrix addition
__global__ void Addition_Kernel(Matrix *Matrix_A, Matrix *Matrix_B, Matrix *C);

// Kernel function for element-wise matrix subtraction
__global__ void Subtraction_Kernel(Matrix *Matrix_A, Matrix *Matrix_B, Matrix *C);

// Kernel function for element-wise matrix subtraction with scalar
__global__ void Subtraction_Kernel(double k, Matrix *Matrix_A, Matrix *Matrix_B);

// Activation function - ReLU
__global__ void ReLU_Activation_Kernel(Matrix *Matrix_A);

// Derivative of activation function - ReLU
__global__ void Derivative_ReLU_Kernel(Matrix *Matrix_A);

// Activation function - Hyperbolic Tangent
__global__ void Tanh_Activation_Kernel(Matrix *Matrix_A);

// Element-wise matrix exponentiation
__global__ void Exponentiation_Kernel(Matrix *Matrix_A);

// Element-wise matrix power
__global__ void Power_Kernel(Matrix *Matrix_A, double b);

// Kernel function for matrix summation over specified axis
__global__ void Sum_Kernel(Matrix *Matrix_A, Matrix *Matrix_B, int axis);

// Kernel function for element-wise matrix division
__global__ void Division_Kernel(Matrix *Matrix_A, Matrix *sum);

// Kernel function for counting elements equal to 1
__global__ void Equal1_Count(Matrix *Matrix_A, Matrix *Matrix_B, int *count);

#endif



