#!/bin/bash
# Run the test executable and redirect its output to test.txt
time ./test > cuda_output.txt

# Run the pytorch.py script and redirect its output to pytorch.txt
time python ./Pytorch/pytorch_train.py > pytorch_output.txt

# Print the execution time for test and pytorch.py to the screen
echo "Execution time for cuda:"
time ./test >/dev/null 2>&1

echo "Execution time for pytorch:"
time python ./Pytorch/pytorch_train.py >/dev/null 2>&1

