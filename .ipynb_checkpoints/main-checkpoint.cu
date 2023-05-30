// Import required libraries
#include <vector>
#include <fstream>
#include <algorithm>
#include "cifar10_reader.cuh" // Import a custom header file
#include "matrix.cuh" // Import a custom header file


// Define some constants and variables
int training_examples = 50000;
int testing_examples = 10000;
int counter, loss_value;
const int batch_size = 64;
double learning_rate = 0.0001;
double regularization_lambda = 0.00;
int number_of_examples, examples_per_epoch, input_dimension, hidden_layer_dimensions[5] = {0, 2048, 1024, 512, 10};

// Declare pointers to custom Matrix class instances
Matrix *X_batch, *Y_batch, *X_test, *Y_test, *predicted_Y, *weights[4], *biases[4], *activation[4], *delta_arr[4], *d_weights[4], *d_biases[4];
Matrix *softmax_sum;

// Declare instances of custom Matrix class
Matrix X_train_data, Y_train_data, X_test_data, Y_test_data;

// Declare functions used in the main code
void prepare_data(); // Prepare the data for training
void predict_y(); // Predict the output values from the input data
void calculate_loss(); // Calculate the loss of the model
void forward_propagation(); // Perform forward propagation
void backward_propagation(); // Perform backward propagation
void train_model(int num_passes, bool print_loss); // Train the model




// This function prepares the data for the neural network training.
void prepare_data() {
    // Prints the input dimension.
    printf("input dim: %d\n", input_dimension);

    // Initialize each matrix and allocate shared memory for them.
    // Initialize X_batch.
    hidden_layer_dimensions[0] = input_dimension;
    cudaMallocManaged((void ** )&(X_batch), sizeof(Matrix));
    X_batch->height = batch_size; X_batch->width = input_dimension;
    cudaMallocManaged((void ** )&(X_batch->elements), batch_size * input_dimension * sizeof(double));
    
    // Initialize Y_batch.
    cudaMallocManaged((void ** )&(Y_batch), sizeof(Matrix));
    Y_batch->height = batch_size; Y_batch->width = 10;
    cudaMallocManaged((void ** )&(Y_batch->elements), batch_size * 10 * sizeof(double));
    
    // Initialize X_test.
    cudaMallocManaged((void ** )&(X_test), sizeof(Matrix));
    X_test->height = batch_size; X_test->width = input_dimension;
    cudaMallocManaged((void ** )&(X_test->elements), batch_size * input_dimension * sizeof(double));
    
    // Initialize Y_test.
    cudaMallocManaged((void ** )&(Y_test), sizeof(Matrix));
    Y_test->height = batch_size; Y_test->width = 10;
    cudaMallocManaged((void ** )&(Y_test->elements), batch_size * 10 * sizeof(double));
    
    // Initialize predicted_Y.
    cudaMallocManaged((void ** )&(predicted_Y), sizeof(Matrix));
    predicted_Y->height = batch_size; predicted_Y->width = 10;
    cudaMallocManaged((void ** )&(predicted_Y->elements), batch_size * 10 * sizeof(double));
    
    // Initialize weights, biases, activation, delta, and d_weights for each of the four hidden layers.
    for (int i = 0; i < 4; i++) {
        int row = hidden_layer_dimensions[i], col = hidden_layer_dimensions[i + 1];
        double std = sqrt(col);
        
        // Initialize weights for the current layer.
        cudaMallocManaged((void ** )&(weights[i]), sizeof(Matrix));
        weights[i]->width = col; weights[i]->height = row;
        cudaMallocManaged((void ** )&(weights[i]->elements), row * col * sizeof(double));
        Initilization(weights[i], std);
        
        // Initialize biases for the current layer.
        cudaMallocManaged((void ** )&(biases[i]), sizeof(Matrix));
        biases[i]->width = col; biases[i]->height = 1;
        cudaMallocManaged((void ** )&(biases[i]->elements), 1 * col * sizeof(double));
        Initilization(biases[i], 0);
        
        // Initialize activation for the current layer.
        cudaMallocManaged((void ** )&(activation[i]), sizeof(Matrix));
        activation[i]->width = col; activation[i]->height = batch_size;
        cudaMallocManaged((void ** )&(activation[i]->elements), batch_size * col * sizeof(double));
        
        // Initialize delta for the current layer.
        cudaMallocManaged((void ** )&(delta_arr[i]), sizeof(Matrix));
        delta_arr[i]->width = col; delta_arr[i]->height = batch_size;
        cudaMallocManaged((void ** )&(delta_arr[i]->elements), batch_size * col * sizeof(double));
        
        // Initialize d_weights for the current layer.
        cudaMallocManaged((void ** )&(d_weights[i]), sizeof(Matrix));
        d_weights[i]->width = col; d_weights[i]->height = row;
        cudaMallocManaged((void ** )&(d_weights[i]->elements), row * col * sizeof(double));
        
        // Initialize d_biases for the current layer.
        cudaMallocManaged((void ** )&(d_biases[i]), sizeof(Matrix));
        d_biases[i]->width = col; d_biases[i]->height = 1;
        cudaMallocManaged((void ** )&(d_biases[i]->elements), row * col * sizeof(double));
        
        // Print the fully connected layers' dimensions.
        printf("fc: %d -> %d\n", row, col);
    }
    
    // Allocate shared memory for the softmax_sum matrix.
    cudaMallocManaged((void ** )&(softmax_sum), sizeof(Matrix));
    softmax_sum->width = 1; softmax_sum->height = batch_size;
    cudaMallocManaged((void ** )&(softmax_sum->elements), batch_size * 1 * sizeof(double));
    
    // Synchronize the device using cudaDeviceSynchronize() to ensure all memory allocations are complete.
    cudaDeviceSynchronize();
}






void predict_y() {
    // Set the number of threads and blocks for parallel processing using image size
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);   
    // Define the layers of the neural network
    // Input layer: Xtest, hidden layers: activation[0], activation[1], activation[2], output layer: activation[3]   
    // Apply the first layer of the neural network to the input data Xtest
    Dot_Product_Kernel <<<gridSize, blockSize>>> (X_test, weights[0], activation[0]);
    // Add the bias term to the first layer and apply ReLU activation function
    Addition_Kernel <<<gridSize, blockSize>>> (activation[0], biases[0], activation[0]);
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[0]);
    // Apply the second layer of the neural network to the output of the first layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[0], weights[1], activation[1]);
    // Add the bias term to the second layer and apply ReLU activation function
    Addition_Kernel <<<gridSize, blockSize>>> (activation[1], biases[1], activation[1]);
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[1]);
    // Apply the third layer of the neural network to the output of the second layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[1], weights[2], activation[2]);
    // Add the bias term to the third layer and apply ReLU activation function
    Addition_Kernel <<<gridSize, blockSize>>> (activation[2], biases[2], activation[2]);
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[2]);
    // Apply the fourth layer of the neural network to the output of the third layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[2], weights[3], activation[3]);
    // Add the bias term to the fourth layer
    Addition_Kernel <<<gridSize, blockSize>>> (activation[3], biases[3], activation[3]);
    // Wait for all kernels to complete execution before proceeding
    cudaDeviceSynchronize();
    // For each input sample, find the predicted label based on the output values in activation[3]
    for (int i = 0; i < activation[3]->height; i++) {
        int Id_max = 0; 
        double Val_max = activation[3]->elements[i * activation[3]->width];
        // Find the index with the highest output value
        for (int j = 0; j < activation[3]->width; j++)
            if (activation[3]->elements[i * activation[3]->width + j] > Val_max) {
                Id_max = j;
                Val_max = activation[3]->elements[i * activation[3]->width + j];
            }
        // If the predicted label matches the true label, increment the count
        if (Y_test->elements[i * activation[3]->width + Id_max])
            counter++;
    }
    // Wait for all kernels to complete execution before returning
    cudaDeviceSynchronize();
}








// This function calculates the loss of the neural network on a batch of data
void calculate_loss() {
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);   
    // Define the layers: Input layer: X_batch, Hidden layers:  activation[0], activation[1], activation[2], Output layer: activation[3]
    // Compute the dot product of the input batch and the weights of the first layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (X_batch, weights[0], activation[0]);
    // Add the biases of the first layer to the dot product
    Addition_Kernel <<<gridSize, blockSize>>> (activation[0], biases[0], activation[0]);
    // Apply ReLU activation function to the first layer
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[0]);
    // Compute the dot product of the first hidden layer and the weights of the second layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[0], weights[1], activation[1]);
    // Add the biases of the second layer to the dot product
    Addition_Kernel <<<gridSize, blockSize>>> (activation[1], biases[1], activation[1]);
    // Apply ReLU activation function to the second layer
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[1]);
    // Compute the dot product of the second hidden layer and the weights of the third layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[1], weights[2], activation[2]);
    // Add the biases of the third layer to the dot product
    Addition_Kernel <<<gridSize, blockSize>>> (activation[2], biases[2], activation[2]);
    // Apply ReLU activation function to the third layer
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[2]);
    // Compute the dot product of the third hidden layer and the weights of the output layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[2], weights[3], activation[3]);
    // Add the biases of the output layer to the dot product
    Addition_Kernel <<<gridSize, blockSize>>> (activation[3], biases[3], activation[3]);
    cudaDeviceSynchronize();
    // Loop over the output layer to calculate the loss
    for (int i = 0; i < activation[3]->height; i++) {
        int Id_max = 0; 
        double Val_max = activation[3]->elements[i * activation[3]->width];
        // Find the index of the maximum value in the output layer
        for (int j = 0; j < activation[3]->width; j++)
            if (activation[3]->elements[i * activation[3]->width + j] > Val_max) {
                Id_max = j;
                Val_max = activation[3]->elements[i * activation[3]->width + j];
            }
        // Increment the loss if the predicted class is incorrect
        if (Y_batch->elements[i * activation[3]->width + Id_max])
            loss_value++;
    }
    cudaDeviceSynchronize();
}




void forward_propagation() {
    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);
    // Define the layers: input layer X_batch, hidden layers activation[0], activation[1], activation[2], output layer activation[3]
    // Perform dot product between X_batch and weights[0] and store the result in activation[0]
    Dot_Product_Kernel <<<gridSize, blockSize>>> (X_batch, weights[0], activation[0]);
    // Add biases[0] to activation[0]
    Addition_Kernel <<<gridSize, blockSize>>> (activation[0], biases[0], activation[0]);
    // Apply ReLU activation function to activation[0]
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[0]);
    // Perform dot product between activation[0] and weights[1] and store the result in activation[1]
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[0], weights[1], activation[1]);
    // Add biases[1] to activation[1]
    Addition_Kernel <<<gridSize, blockSize>>> (activation[1], biases[1], activation[1]);
    // Apply ReLU activation function to activation[1]
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[1]);
    // Perform dot product between activation[1] and weights[2] and store the result in activation[2]
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[1], weights[2], activation[2]);
    // Add biases[2] to activation[2]
    Addition_Kernel <<<gridSize, blockSize>>> (activation[2], biases[2], activation[2]);
    // Apply ReLU activation function to activation[2]
    ReLU_Activation_Kernel <<<gridSize, blockSize>>> (activation[2]);
    // Perform dot product between activation[2] and weights[3] and store the result in activation[3]
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[2], weights[3], activation[3]);
    // Add biases[3] to activation[3]
    Addition_Kernel <<<gridSize, blockSize>>> (activation[3], biases[3], activation[3]);
    // Apply softmax function to activation[3]
    Exponentiation_Kernel <<<gridSize, blockSize>>> (activation[3]);
    // Compute the sum of all elements of activation[3] and store it in softmax_sum
    Sum_Kernel <<<gridSize, blockSize>>> (activation[3], softmax_sum, 1);
    // Divide each element of activation[3] by softmax_sum
    Division_Kernel <<<gridSize, blockSize>>> (activation[3], softmax_sum);
    // Wait for all CUDA threads to finish
    cudaDeviceSynchronize();
}













//backPropagation function
void backward_propagation() {
    //Set up block size and grid size for CUDA kernel
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);
    //Backpropagation
    //Calculate derivative of the cost function with respect to the output layer activation
    Subtraction_Kernel <<<gridSize, blockSize>>> (activation[3], Y_batch, delta_arr[3]);
    cudaDeviceSynchronize();

    //Calculate the gradients for the weights and biases of the output layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[2], delta_arr[3], d_weights[3], true);   
    Sum_Kernel <<<gridSize, blockSize>>> (delta_arr[3], d_biases[3], 0);
    cudaDeviceSynchronize();

    //Calculate derivative of the output layer activation with respect to the input of the last hidden layer
    Derivative_ReLU_Kernel <<<gridSize, blockSize>>> (activation[2]);
    //Calculate the product of the derivative of the cost function with respect to the output layer activation and the derivative of the output layer activation with respect to the input of the last hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (delta_arr[3], weights[3], delta_arr[2], false, true);      
    Multiplication_Kernel <<<gridSize, blockSize>>> (delta_arr[2], activation[2], delta_arr[2]);
    //Calculate the gradients for the weights and biases of the last hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[1], delta_arr[2], d_weights[2], true);
    Sum_Kernel <<<gridSize, blockSize>>> (delta_arr[2], d_biases[2], 0);
    cudaDeviceSynchronize();

    //Calculate derivative of the last hidden layer activation with respect to the input of the second last hidden layer
    Derivative_ReLU_Kernel <<<gridSize, blockSize>>> (activation[1]);
    //Calculate the product of the derivative of the cost function with respect to the output layer activation and the derivative of the output layer activation with respect to the input of the last hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (delta_arr[2], weights[2], delta_arr[1], false, true);      
    Multiplication_Kernel <<<gridSize, blockSize>>> (delta_arr[1], activation[1], delta_arr[1]);
    //Calculate the gradients for the weights and biases of the second last hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (activation[0], delta_arr[1], d_weights[1], true);
    Sum_Kernel <<<gridSize, blockSize>>> (delta_arr[1], d_biases[1], 0);
    cudaDeviceSynchronize();

    //Calculate derivative of the second last hidden layer activation with respect to the input of the first hidden layer
    Derivative_ReLU_Kernel <<<gridSize, blockSize>>> (activation[0]);
    //Calculate the product of the derivative of the cost function with respect to the output layer activation and the derivative of the output layer activation with respect to the input of the last hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (delta_arr[1], weights[1], delta_arr[0], false, true);      
    Multiplication_Kernel <<<gridSize, blockSize>>> (delta_arr[0], activation[0], delta_arr[0]);
    //Calculate the gradients for the weights and biases of the first hidden layer
    Dot_Product_Kernel <<<gridSize, blockSize>>> (X_batch, delta_arr[0], d_weights[0], true);
    Sum_Kernel <<<gridSize, blockSize>>> (delta_arr[0], d_biases[0], 0);
    cudaDeviceSynchronize();

    //Gradient update
    //Update the weights and biases of all layers using the calculated gradients and the learning rate
    for (int i = 0; i < 4; i++) {
        Multiplication_Kernel <<<gridSize, blockSize>>> (d_weights[i], learning_rate);
        Multiplication_Kernel <<<gridSize, blockSize>>> (d_biases[i], learning_rate);
        Subtraction_Kernel <<<gridSize, blockSize>>> (weights[i], d_weights[i], weights[i]);
        Subtraction_Kernel <<<gridSize, blockSize>>> (biases[i], d_biases[i], biases[i]);
        cudaDeviceSynchronize();
    }
}





//Function to train the model with given number of passes and option to print loss
void train_model(int nb_of_pass, bool printLoss) {
    std::ofstream file1;
    std::ofstream file2;
    file1.open("./statistics_train.csv", std::ios::app);
    file2.open("./statistics_test.csv", std::ios::app);
    
    int i;
    //Loop through all the passes
    for (i = 0; i <= nb_of_pass; i++) {
        //Get the index of current example in the epoch
        int j = i % examples_per_epoch;
        //Shuffle the training set after each complete training iteration
        if (j == 0) {
            //Create a vector of indices
            vector<int> ridx(number_of_examples);
            int k;
            //Fill the vector with indices
            for (k = 0; k < number_of_examples; k++)
                ridx[k] = k;
            //Shuffle the vector of indices
            random_shuffle(ridx.begin(), ridx.end());
            //Shuffle the training data based on shuffled indices
            X_train_data.shuffle(ridx);
            Y_train_data.shuffle(ridx);
        }
        //Get a batch from the training set
        Copy_Data(X_batch, X_train_data, j * batch_size, (j + 1) * batch_size);
        Copy_Data(Y_batch, Y_train_data, j * batch_size, (j + 1) * batch_size, true);
        //Perform forward propagation
        forward_propagation();
        //Perform backward propagation
        backward_propagation();
        //If option to print loss is enabled and current iteration is a multiple of 100
        if (printLoss && (i % 100 == 0)) {
            //Decrease learning rate
            learning_rate *= 0.99;
            counter = 0;
            //Split the test set into batches and predict them one by one
            for (int k = 0; k < (int)(X_test_data.height / batch_size); k++) {
                Copy_Data(X_test, X_test_data, k * batch_size, (k + 1) * batch_size);
                Copy_Data(Y_test, Y_test_data, k * batch_size, (k + 1) * batch_size, true);   
                predict_y();     
                cudaDeviceSynchronize();
            }
            //Calculate test set accuracy
            double accuracy = (counter * 1.0 / X_test_data.height);
            //Get current time
            struct tm *p;
            time_t t = time(0);
            p = localtime(&t);
                 // write header if file is empty
           std::ifstream ifile("./statistics_test.csv");
           if (ifile.peek() == std::ifstream::traits_type::eof()) {
             file2 <<"iter,test_acc"<< std::endl;
           }
           file2 <<i <<","<< accuracy*100<< std::endl;
            //Print testing accuracy
            printf("%02d:%02d:%02d The testing accuracy after iteration %d is: %.2lf%%\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);

        }

        //After completing a full test set, output the train loss
        if (printLoss && (j == 0) && (i != 0)) {
            //Calculate train set loss
            double accuracy = (loss_value * 1.0 / X_train_data.height);
            //Get current time
            struct tm *p;
            time_t t = time(0);
            p = localtime(&t);
            std::ifstream ifile("./statistics_train.csv");
            if (ifile.peek() == std::ifstream::traits_type::eof()) {
             file1 <<"iter,train_acc"<< std::endl;
            }
            file1 <<i<<","<<accuracy*100<< std::endl;
            //Print train loss
            printf("\n%02d:%02d:%02d The train accuracy after iteration %d is: %.2lf%%\n\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);
            //Reset loss value for next epoch
            loss_value = 0;
        }
        
        //Calculate loss for the current batch
        calculate_loss();
    }
}






// Main function
int main() {
    cout.precision(16); // Set the precision of console output to 16 decimal places

    // Read in the CIFAR-10 dataset using a custom function and store it in a map
    unordered_map<string, Matrix> dataMap = readCIFAR10Data();

    // Extract the training and testing data and labels from the map
    X_train_data = dataMap["trainImages"];
    Y_train_data = dataMap["trainLabels"];
    X_test_data = dataMap["testImages"];
    Y_test_data = dataMap["testLabels"];

    // Print out the dimensions of the training and testing data and labels
    printf("(%d, %d) (%d, %d)\n", X_train_data.height, X_train_data.width, Y_train_data.height, Y_train_data.width);
    printf("(%d, %d) (%d, %d)\n", X_test_data.height, X_test_data.width, Y_test_data.height, Y_test_data.width);

    // Set some necessary variables based on the dimensions of the data
    number_of_examples = X_train_data.height;
    input_dimension = X_train_data.width;
    examples_per_epoch = (int)(number_of_examples / batch_size);

    // Prepare the data for training
    prepare_data();

    // Train the model using 5000 passes and print the loss every epoch
    train_model(5000, true);

    return 0; 
}


