#include "common.h"
#include "binaryHeapSeq.h"

#include <bits/stdc++.h>
#include <ctime>
#include <assert.h>
#include <unistd.h>

using namespace std;

// Global variables
int *Arr;            
int *receivedArr;    
int *binaryArrayOutput;

Heap *d_Heap;
Partial_Buffer *d_partialBuffer;
int *d_heapLock;

// Function declarations
void computeRandomInputArray(int *Arr, int n);
void verifyResults(int *h_arr, int *d_arr, int n);
void benchmark();

int main() {
    benchmark(); 
}

// Function definitions
void computeRandomInputArray(int *Arr, int n) {
    int m = 1e8;
    for (int i = 0; i < n; i++)
        Arr[i] = rand() % m;
}

void verifyResults(int *h_arr, int *d_arr, int n) {
    bool correct = 1;
    for (int i = 0; i < n; i++) {
        if (h_arr[i] != d_arr[i]) {
            correct = 0;
            cout << h_arr[i] << " " << d_arr[i] << " " << i << "\n";
            break;
        }
    }
    cout << ((correct) ? "Success\n" : "Failed!\n");
}

void benchmark() {
    int n = NUMBER_OF_NODES;
int heapCapacity = n * BATCH_SIZE;
Arr = new int[heapCapacity];
binaryArrayOutput = new int[heapCapacity];
receivedArr = new int[heapCapacity];
computeRandomInputArray(Arr, heapCapacity);

cout << "Running on GPU...\n";
std::clock_t cStartGPU = std::clock();
initializeHeap();
int *d_arr;
cudaMalloc((void **)&d_arr, heapCapacity * sizeof(int));
cudaMemcpy(d_arr, Arr, heapCapacity * sizeof(int), cudaMemcpyHostToDevice);

int *d_arrRec;
cudaMalloc((void **)&d_arrRec, heapCapacity * sizeof(int));

std::clock_t cStartInsertGPU = std::clock();
for (int i = 0; i < n; i++) {
    insertKeys(d_arr + i * BATCH_SIZE, BATCH_SIZE);
}
cudaDeviceSynchronize();
std::clock_t cEndInsertGPU = std::clock();
long double timeElapsedInsertGPU = 1000.0 * (cEndInsertGPU - cStartInsertGPU) / CLOCKS_PER_SEC;
std::cout << "GPU insertions Execution time : \t" << timeElapsedInsertGPU << " ms\n";

std::clock_t cStartDeleteGPU = std::clock();
for (int i = 0; i < n; i++) {
    deleteKeys(d_arrRec + i * BATCH_SIZE, n * BATCH_SIZE / 2);
}
cudaDeviceSynchronize();
std::clock_t cEndDeleteGPU = std::clock();
long double timeElapsedDeleteGPU = 1000.0 * (cEndDeleteGPU - cStartDeleteGPU) / CLOCKS_PER_SEC;
std::cout << "GPU deletions Execution time : \t" << timeElapsedDeleteGPU << " ms\n";

long double timeElapsedGPU = timeElapsedInsertGPU + timeElapsedDeleteGPU;
std::cout << "GPU Total Execution time :\t" << timeElapsedGPU << " ms\n";

cudaMemcpy(receivedArr, d_arrRec, heapCapacity * sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(d_arr);
cudaFree(d_arrRec);
heap_finalise();

cout << "\nRunning on CPU (Binary Heap)...\n";
// Binary Heap benchmarking
BinaryHeap binaryHeapSeq(heapCapacity);
std::clock_t cStartInsertCPU = std::clock();
for (int i = 0; i < n; i++) {
    for (int j = i * BATCH_SIZE; j < (i + 1) * BATCH_SIZE; j++) {
        binaryHeapSeq.insert(Arr[j]);
    }
}
std::clock_t cEndInsertCPU = std::clock();
long double timeElapsedInsertCPU = 1000.0 * (cEndInsertCPU - cStartInsertCPU) / CLOCKS_PER_SEC;
std::cout << "CPU insertions Execution time:\t" << timeElapsedInsertCPU << " ms\n";

std::clock_t cStartDeleteCPU = std::clock();
int x = 0;
while (!binaryHeapSeq.isEmpty()) {
    binaryArrayOutput[x++] = binaryHeapSeq.deleteMin();
}
std::clock_t cEndDeleteCPU = std::clock();
long double timeElapsedDeleteCpu = 1000.0 * (cEndDeleteCPU - cStartDeleteCPU) / CLOCKS_PER_SEC;
std::cout << "CPU deletions Execution time:\t" << timeElapsedDeleteCpu << " ms\n";

long double timeElapsedMsCpu = timeElapsedInsertCPU + timeElapsedDeleteCpu;
std::cout << "CPU Total Execution time:\t" << timeElapsedMsCpu << " ms\n";

    // Verify
    sort(Arr, Arr + heapCapacity);
    cout << "\nVerifying GPU output...\n";
    verifyResults(Arr, receivedArr, heapCapacity);
    cout << "\nVerifying CPU output...\n";
    verifyResults(Arr, binaryArrayOutput, heapCapacity);

    cout << "\nRunning InsertsDeletes(Interleaved) benchmark on GPU...\n";
    std::clock_t cStartGPUInsertDelete = std::clock();

    initializeHeap();

    cudaMalloc((void **)&d_arr, heapCapacity * sizeof(int));
    cudaMemcpy(d_arr, Arr, heapCapacity * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_arrRec, heapCapacity * sizeof(int));

    std::clock_t cStartInsertDeleteCPU = std::clock();

    srand(time(NULL));
    int interleavingFactor = rand() % 10 + 1; 
    insertAndDeleteKeys(d_arr, n * BATCH_SIZE / 2, d_arrRec, n * BATCH_SIZE / 2, interleavingFactor);

    cudaDeviceSynchronize();

    std::clock_t cEndInsertDeleteCPU = std::clock();
    long double timeMsInsertDeleteInterleavedCPU = 1000.0 * (cEndInsertDeleteCPU - cStartInsertDeleteCPU) / CLOCKS_PER_SEC;
    std::cout << "GPU InsertsDeletes(Interleaved) Execution time : \t" << timeMsInsertDeleteInterleavedCPU << " ms\n";

    cudaMemcpy(receivedArr, d_arrRec, heapCapacity * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_arrRec);
    heap_finalise();

    std::clock_t cEndGPUInsertDelete = std::clock();
    timeMsInsertDeleteInterleavedCPU = 1000.0 * (cEndGPUInsertDelete - cStartGPUInsertDelete) / CLOCKS_PER_SEC;
    std::cout << "GPU InsertsDeletes(Interleaved) Full Execution time :\t" << timeMsInsertDeleteInterleavedCPU << " ms\n";

    cout << "\nRunning InsertsDeletes(Interleaved) benchmark on CPU (Binary Heap)...\n";

    // Binary Heap benchmarking
    BinaryHeap binaryHeapSeq_InsertDelete(heapCapacity);
    cStartInsertDeleteCPU = std::clock();

    for (int i = 0; i < heapCapacity / 2; i++) {
        binaryHeapSeq_InsertDelete.insert(Arr[i]);
    }

    srand(time(NULL));
    interleavingFactor = rand() % 10 + 1; 
    int insertCount = heapCapacity / 2;
    int deleteCount = 0;
    while (insertCount < heapCapacity || !binaryHeapSeq_InsertDelete.isEmpty()) {
        if (insertCount < heapCapacity && insertCount - deleteCount < interleavingFactor) {
            int value = Arr[insertCount];
            binaryHeapSeq_InsertDelete.insert(value);
            insertCount++;
        } else if (!binaryHeapSeq_InsertDelete.isEmpty()) {
            binaryArrayOutput[deleteCount++] = binaryHeapSeq_InsertDelete.deleteMin();
        } else {
            break;
        }
    }

    cEndInsertDeleteCPU = std::clock();
    timeMsInsertDeleteInterleavedCPU = 1000.0 * (cEndInsertDeleteCPU - cStartInsertDeleteCPU) / CLOCKS_PER_SEC;
    std::cout << "CPU InsertsDeletes(Interleaved) Execution time:\t" << timeMsInsertDeleteInterleavedCPU << " ms\n";
}