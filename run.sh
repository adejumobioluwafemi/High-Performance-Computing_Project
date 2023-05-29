#!/bin/bash

# Run the "test" executable and record its execution time
start_time=$(date +%s.%N)
./test
test_time=$(echo "$(date +%s.%N) - $start_time" | bc)

# Run the python script and record its execution time
start_time=$(date +%s.%N)
python pytorch.py
py_time=$(echo "$(date +%s.%N) - $start_time" | bc)

# Output the execution times
echo "Execution time for test: $test_time seconds"
echo "Execution time for python script: $py_time seconds"
