
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

:'
#!/bin/bash

# Run the test executable and redirect its output to test.txt
time ./test > test.txt

# Run the pytorch.py script and redirect its output to pytorch.txt
time python pytorch.py > pytorch.txt

# Print the execution time for test and pytorch.py to the screen
echo "Execution time for test:"
time ./test >/dev/null 2>&1

echo "Execution time for pytorch.py:"
time python pytorch.py >/dev/null 2>&1
'
