# Generalized Heap Implementation on CPU

NOTE: This code is a sequential implementation of the heap inspired by the paper ["Accelerating Concurrent Heap on GPUs"](https://arxiv.org/pdf/1906.06504.pdf) by Yanhao Chen et al. Due to my weak c/cuda skills at the beginning of this project, my implementation, while giving a correct output, is too slow to use as a benchmark for the parallel versions. Hence, the parallel versions are verified using a binary heap implementation, different from this one. I did not go back to fix this code at the end because I focused on the parallel optimized version for this project.

## How to Use

### Compilation

To compile the project, run the following command in your terminal:

make all


### Execution

To execute the program, use the following command format:

./main.exe input.txt output.txt 20 100


- `input.txt`: This file should contain numbers separated by spaces (e.g., `4 3 2 1`). This file is a required argument, and you can use any input file name as long as the file exists in the correct directory.
- `output.txt`: This file will be created by the program to store the output. If the file does not exist, it will be created. You can specify any name for the output file.
- `20` and `100` are optional arguments representing the levels and node capacity, respectively. Their default values are `20` and `100`. 

### Notes

- If during use, you enter input sizes that require more memory than allocated, the `output.txt` file will be empty. In such cases, it is advisable to increase the node capacity. For example, ./main.exe input.txt output.txt 20 200
- We recommend keeping the levels at `20`, as this configuration has been tested extensively.
- The program functions by inserting the numbers provided in the `input.txt` file, followed by extract-min operations at the end on all of the heap's keys. This approach ensures consistent benchmarking across multiple datasets.
