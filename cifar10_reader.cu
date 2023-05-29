
// Header file containing function definitions for reading CIFAR-10 dataset
#include "cifar10_reader.cuh"

// Function to read CIFAR-10 images from binary file
void readCIFAR10Images(string fileName, Matrix &labels, Matrix &images, int len, int pos) {
    // Open binary file for reading
    ifstream file(fileName, ios::binary);
    if (!file.is_open())
        cout << "Error Opening File!\n" << endl;

    // Loop over each image in the file
    for (int i = 0; i < len; i++) {
        unsigned char pixel = 0;
        // Read label for the current image
        file.read((char *)&pixel, sizeof(pixel));
        labels.elements[i + pos] = (double)pixel;
        // Read pixel values for the current image
        for (int j = 0; j < 3072; j++) {
            file.read((char *)&pixel, sizeof(pixel));
            images.elements[(i + pos) * 3072 + j] = (double)pixel / 255;
        }
    }
    // Close the file
    file.close();
}

// Function to read the entire CIFAR-10 dataset
unordered_map<string, Matrix> readCIFAR10Data() {
    // Declare and initialize variables to store dataset sizes
    extern int training_examples;
    extern int testing_examples;

    // Create matrices to store training and testing data
    Matrix trainImages = Matrix(training_examples, 32 * 32 * 3);
    Matrix trainLabels = Matrix(training_examples, 1);
    Matrix testImages = Matrix(testing_examples, 32 * 32 * 3);
    Matrix testLabels = Matrix(testing_examples, 1);

    // Read training data from binary files
    readCIFAR10Images("./data/data_batch_1.bin", trainLabels, trainImages, 10000, 0);
    readCIFAR10Images("./data/data_batch_2.bin", trainLabels, trainImages, 10000, 10000);
    readCIFAR10Images("./data/data_batch_3.bin", trainLabels, trainImages, 10000, 20000);
    readCIFAR10Images("./data/data_batch_4.bin", trainLabels, trainImages, 10000, 30000);
    readCIFAR10Images("./data/data_batch_5.bin", trainLabels, trainImages, 10000, 40000);
    
    // Read testing data from binary file
    readCIFAR10Images("./data/test_batch.bin", testLabels, testImages, testing_examples, 0);

    // Create an unordered map to store the data matrices
    unordered_map<string, Matrix> dataMap;

    // Insert training and testing data matrices into the map
    dataMap.insert({"trainImages", trainImages});
    dataMap.insert({"trainLabels", trainLabels});
    dataMap.insert({"testImages", testImages});
    dataMap.insert({"testLabels", testLabels});

    // Return the data map
    return dataMap;
}