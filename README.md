# Generalized Heap Implementation on GPUs

This code is based on the concepts presented in the paper ["Accelerating Concurrent Heap on GPUs"](https://arxiv.org/pdf/1906.06504.pdf) by Yanhao Chen et al. Our project aims to optimize heap operations on GPU architectures. The implementation is divided into four distinct strategies, identified as kernels:
0 - Base parallel implementation
1 - Shared memory implementation
2 - Bitonic sort implementation
3 - Early breaking and other optimizations

## How to Use

### Compilation

Navigate to the directory of the desired kernel and run `make`.

### Execution

Execute the program by running `./main`. For Google Colab, execute directly. For HPC environments, refer to the specific documentation to allocate resources.

### Validation and Verification

To verify the correctness of our parallel heap implementations, we use a sequential Binary Heap running on the CPU as a reference model. Kernels 2 and 3 have been validated against large input sizes and show correct results. However, some discrepancies are noted for kernels 0 and 1 with large inputs, attributed to the focus on the optimized kernels.

### Limitations

The input size for the heap must be a power of 2, necessitated by the Bitonic Sort implementation, which requires the data within each node to align with this constraint.

### Thanks and Acknowledgements

Special thanks to Dr. Izzat for his insights on software parallelization techniques and to Dr. Mouawad for his guidance on optimizing data structures for research purposes. Appreciation is also extended to the authors of the referenced paper and to the community whose shared contributions have inspired this project.
