#ifndef COMMON_H
#define COMMON_H

#include <iostream>

// Constant definitions
#define BLOCK_SIZE BATCH_SIZE
#define PARTIAL_BUFFER_CAPACITY (BATCH_SIZE - 1)  
#define HEAP_CAPACITY (NUMBER_OF_NODES + 1) * (BATCH_SIZE)
#define ROOT_NODE_IDX 1
#define MASTER_THREAD 0
#define BATCH_SIZE 1024       
#define NUMBER_OF_NODES (1<<8) 

// Enum for lock states
enum LOCK_STATES {
    AVAILABLE,
    INUSE
};

// Partial buffer structure
struct Partial_Buffer {
    int itemCount = 0;
    int arr[PARTIAL_BUFFER_CAPACITY];
};

// Heap structure
struct Heap {
    int itemCount = 0;
    int arr[HEAP_CAPACITY];
};

// Global variables
extern Heap *d_Heap;
extern Partial_Buffer *d_partialBuffer;
extern int *d_heapLock;

// Function declarations

// Host functions
__host__ void initializeHeap();
__host__ void insertKeys(int *itemsToInsert, int itemsToInsertSize);
__host__ void deleteKeys(int* deletedItems, int deletedItemsSize);
__host__ void heap_finalise();
__host__ void insertAndDeleteKeys(int* itemsToInsert, int itemsToInsertSize, int* deletedKeys, int deletedKeysSize, int interleavingFactor);

// Device functions
__device__ void acquireLock(int *lock, int initialState, int finalState);
__device__ void releaseLockAtomic(int *lock, int initialState, int finalState);
__global__ void initializeHeap(Heap *heap, Partial_Buffer *partialBuffer);
__device__ int reverseBits(int n);
__device__ void copyArray(int *arr1, int startIndex1, int *arr2, int startIndex2, int elementCount);
__device__ void memsetArray(int *arr, int arrStartIndex, int val, int elementCount);
__device__ void sortBubble(int *arr, int size);
__device__ int binarySearch(int *arr1, int high, int search, bool considerEquality);
__device__ void mergeAndSortarrays(int *arr1, int idx1, int *arr2, int idx2, int *mergedarr);
__global__ void topDownInsertion(int *itemsToInsert, int itemCount, int *heapLocks, Partial_Buffer *partialBuffer, Heap *heap, int *mergedResult);
__global__ void topDownDeletion(int* deletedItems, int* heapLocks, Partial_Buffer* partialBuffer, Heap* heap, int* array1, int* array2, int* array3, int* mergedResult);

#endif  