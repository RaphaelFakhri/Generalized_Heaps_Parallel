#include "common.h"



__global__ void initializeHeap(Heap *heap, Partial_Buffer *partialBuffer) {
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if(index == 0) {
        // Only use one thread to set the attributes.
        heap -> itemCount = 0;
        partialBuffer -> itemCount = 0;
    }

    // Set Nodes.
    if (index < HEAP_CAPACITY) {
        heap -> arr[index] = INT_MAX;
    }
    // Set Partial Buffer.
    if (index < PARTIAL_BUFFER_CAPACITY) {
        partialBuffer -> arr[index] = INT_MAX;   
    }
}

// First Optimization: Use parallel sorting, this implementation is based on bitonic sort, not radix sort which we had envisioned.
// Performance improvements from using multiple threads, but limited to using power of 2 elements.
// Time complexity: O(log^2(n)) for n elements.
// Space complexity: O(n).
__device__ void sortBitonic(int *arr, int itemCount) {
    int threadId = threadIdx.x;
    int maximum = 0, minimum = 0, otherId = 0, i = 2, j = 2;

    int batchNumber = threadId >> 1; // Parity of batchNumber determines which threads are active during the cycle (half).

    // outer loop: controls size of sorting array. As it increases, we obtain a bitonic sequence and then a sorted array.
    for (i = 2; i <= itemCount ; i <<= 1, batchNumber >>= 1) {
        // middle loop: performs bitonic swaps for chunks of bitonic sequences.
        for (j = i; j >= 2 ; j >>= 1) {
            int lookAheadSteps = j >> 1;
            if ((threadId % j < lookAheadSteps) && (threadId < itemCount)) 
            {
                otherId = threadId + lookAheadSteps;
                minimum = min(arr[threadId], arr[otherId]);
                maximum = max(arr[threadId], arr[otherId]); 
                if (batchNumber & 1) {
                    arr[threadId] = maximum;
                    arr[otherId] = minimum;
                }
                else {
                    arr[threadId] = minimum;
                    arr[otherId] = maximum;
                }
            }
            __syncthreads();
        }
    }
}

// Parallel array copy algorithm to utilize multiple threads. 
// Not considered an optimization because it is fairly straightforward.
// Time Complexity: O(n)
__device__ void copyArray(int *arr1, int startIndex1, int *arr2, int startIndex2, int elementCount) {
    int threadId = threadIdx.x;
    if (threadId < elementCount) {
        arr2[startIndex2 + threadId] = arr1[startIndex1 + threadId];
    }
    __syncthreads();
}


// Parallel version of the memset function, needed to reset the values of all keys inside a node at once (for example when deleting a node).
// Not considered an optimization because it is fairly straightforward.
// Time Complexity: O(n)

__device__ void memsetArray(int *arr, int arrStartIndex, int val, int elementCount) {
    // sets values of arr to val between given indices of arr
    int threadId = threadIdx.x;
    if (threadId < elementCount) {
        arr[arrStartIndex + threadId] = val;
    }
    __syncthreads();
}

// Bit reversal is useful to create a root to target node path.
// Time Complexity: O(n).
__device__ int reverseBits(int n, int level) {

    if (n <= 4) 
        return n;

    int ans = 1 << (level--);
    while(n != 1) {
        ans += ((n & 1) << (level--));
        n >>= 1;
    }
    return ans;
}

// Lock Management functions, instead of locking the entire heap, we lock individual nodes.
// Limitation: Since we always lock the root node and it is always accessed, we are effectively locking the entire heap.

// Acquire a lock if available.
__device__ void acquireLock(int *lock, int initialState, int finalState) {
    while (atomicCAS(lock, initialState, finalState) != initialState);
}

// Make node available.
__device__ void releaseLockAtomic(int *lock, int initialState, int finalState) {
    atomicCAS(lock, initialState, finalState);
}

// Binary search algorithm to find the index of the smallest element larger than the element searched in arr1 (to maintain heap property).
// Time Complexity: O(log(n)).

__device__ int binarySearch(int *arr1, int high, int search, bool considerEquality) {
    // Boundary conditions
    if(high == 0) return 0;
    int low = 0, mid = 0;
    int ans = high;
    while (low <= high)
    {
        mid = (low + high) >> 1;
        if (arr1[mid] >= search and considerEquality) { // Higher or Equal? LeftChild half
            ans = mid;
            high = mid - 1;
        }
        else if (arr1[mid] > search) { // Strictly Higher? LeftChild half
            ans = mid;
            high = mid - 1;
        }
        else { // Lower? RightChild half
            low = mid + 1;
        }
    }
    return ans;
}

// Merge and sort two arrays in parallel, used to merge nodes from root to target node to preserve heap property.
// Not considered an optimization because it is quite necessary and clearly explained in the paper.
// Time Complexity: O(n log(n)) (due to the binary search), where n is the larger of the two arrays.
// Space Complexity: O(n).

__device__ void mergeAndSortArrays(int *arr1, int idx1, int *arr2, int idx2, int *mergedArr) {

    __syncthreads();
    // Special cases
    if(idx1 == 0) {
        copyArray(arr2, 0, mergedArr, 0, idx2);
    }
    else if(idx2 == 0) {
        copyArray(arr1, 0, mergedArr, 0, idx1);
    }
    
    else if(arr1[idx1 - 1] <= arr2[0]) {
        copyArray(arr1, 0, mergedArr, 0, idx1);
        copyArray(arr2, 0, mergedArr, idx1, idx2);
    }

    else if(arr2[idx2 - 1] <= arr1[0]) {\
        copyArray(arr2, 0, mergedArr, 0, idx2);
        copyArray(arr1, 0, mergedArr, idx2, idx1);
    }
    // General case
    else {

        int threadId = threadIdx.x;
        if (threadId < idx1) {
            int x = binarySearch(arr2, idx2, arr1[threadId], 1);
            mergedArr[threadId + x] = arr1[threadId];
        }

        if (threadId < idx2) {
            int x = binarySearch(arr1, idx1, arr2[threadId], 0);
            mergedArr[threadId + x] = arr2[threadId];
        }
    }
    __syncthreads();
    
}

__global__ void topDownInsertion(int *itemsToInsert, int itemCount, int *heapLocks, Partial_Buffer *partialBuffer, Heap *heap, int *mergedResult) {
    int threadId = threadIdx.x;

    int *insertionItems = itemsToInsert;
    sortBitonic(insertionItems, itemCount);

    if (threadId == MASTER_THREAD){
        acquireLock(&heapLocks[ROOT_NODE_IDX], AVAILABLE, INUSE);
    }
    __syncthreads();

    int combinedItemCount = partialBuffer->itemCount + itemCount;

    mergeAndSortArrays(insertionItems, itemCount, partialBuffer->arr, partialBuffer->itemCount, mergedResult);

    // Check if the combined item count exceeds the batch size
    if (combinedItemCount >= BATCH_SIZE) {
        copyArray(mergedResult, 0, insertionItems, 0, BATCH_SIZE);
        copyArray(mergedResult, BATCH_SIZE , partialBuffer->arr, 0, combinedItemCount - BATCH_SIZE);
        __threadfence();

        if (threadId == MASTER_THREAD)
            atomicExch(&(partialBuffer->itemCount), combinedItemCount - BATCH_SIZE);
        __syncthreads();
    } else {
        if (heap->itemCount == 0) {
            copyArray(mergedResult, 0, partialBuffer->arr, 0, combinedItemCount);
            __threadfence();
        } else {
            copyArray(mergedResult, 0, heap->arr, ROOT_NODE_IDX * BATCH_SIZE, BATCH_SIZE);
            __syncthreads();

            mergeAndSortArrays(heap->arr, ROOT_NODE_IDX * BATCH_SIZE, insertionItems, BATCH_SIZE, mergedResult);

            copyArray(mergedResult, 0, heap->arr, ROOT_NODE_IDX * BATCH_SIZE, BATCH_SIZE);
            __threadfence();

            copyArray(mergedResult, BATCH_SIZE, partialBuffer->arr, 0, combinedItemCount);
            __threadfence();
        }
        if (threadId == MASTER_THREAD)
            partialBuffer->itemCount = combinedItemCount;

        __syncthreads();

        if (threadId == MASTER_THREAD)
            releaseLockAtomic(&heapLocks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        return;
    }

    // Update the heap item count
    if (threadId == MASTER_THREAD)
        atomicAdd(&(heap->itemCount), 1);
    __syncthreads();

    // Calculate the target node and its level in the heap
    int targetNode = heap->itemCount, level = -1;
    int tempTargetNode = targetNode;
    while(tempTargetNode) {
        level++;
        tempTargetNode >>= 1;
    }

    // Reverse the bits to find the correct position in the heap
    targetNode = reverseBits(targetNode, level);
    
    // Lock management to avoid deadlocks and ensure thread synchronization
    if (targetNode != ROOT_NODE_IDX) {
        if (threadId == MASTER_THREAD) {
            acquireLock(&heapLocks[targetNode], AVAILABLE, INUSE);
        }
        __syncthreads();
    }

    int low = 0, currentNode = ROOT_NODE_IDX;
    while (currentNode != targetNode) {
        low = currentNode * BATCH_SIZE;
       
        copyArray(heap->arr, low, mergedResult, 0, BATCH_SIZE);

        mergeAndSortArrays(mergedResult, BATCH_SIZE, insertionItems, BATCH_SIZE, mergedResult);

        copyArray(mergedResult, 0, heap->arr, low, BATCH_SIZE);
        __threadfence();

        copyArray(mergedResult, BATCH_SIZE, insertionItems, 0, BATCH_SIZE);

        currentNode = targetNode >> (--level);

        if (threadId == MASTER_THREAD) {
            if (currentNode != targetNode) {
                acquireLock(&heapLocks[currentNode], AVAILABLE, INUSE);
            }
            releaseLockAtomic(&heapLocks[currentNode >> 1], INUSE, AVAILABLE);
        }
        __syncthreads();
    }

    copyArray(insertionItems, 0, heap->arr, targetNode * BATCH_SIZE, BATCH_SIZE);
    __threadfence();
    if (threadId == MASTER_THREAD) {
        releaseLockAtomic(&heapLocks[targetNode], INUSE, AVAILABLE);
    }
    __syncthreads();
}


__global__ void topDownDeletion(int* deletedItems, int* heapLocks, Partial_Buffer* partialBuffer, Heap* heap, int* array1, int* array2, int* array3, int* mergedResult) {
    int threadId = threadIdx.x;

    if (threadId == MASTER_THREAD) {
        acquireLock(&heapLocks[ROOT_NODE_IDX], AVAILABLE, INUSE);
    }
    __syncthreads();

    if (heap->itemCount == 0) {
        if (partialBuffer->itemCount != 0) {
            copyArray(partialBuffer->arr, 0, deletedItems, 0, partialBuffer->itemCount);
            __threadfence();
            if (threadId == MASTER_THREAD) {
                atomicExch(&(partialBuffer->itemCount), 0);
            }
            __syncthreads();
        }
        if (threadId == MASTER_THREAD) {
            releaseLockAtomic(&heapLocks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        }
        return;
    }

    copyArray(heap->arr, ROOT_NODE_IDX * BATCH_SIZE, deletedItems, 0, BATCH_SIZE);
    memsetArray(heap->arr, ROOT_NODE_IDX * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();

    int targetNode = heap->itemCount, level = -1;
    int tempTargetNode = targetNode;
    while (tempTargetNode) {
        level++;
        tempTargetNode >>= 1;
    }
    targetNode = reverseBits(targetNode, level);
    __syncthreads();

    if (threadId == MASTER_THREAD) {
        atomicAdd(&(heap->itemCount), -1);
    }
    __syncthreads();

    if (targetNode == 1) {
        if (threadId == MASTER_THREAD) {
            releaseLockAtomic(&heapLocks[ROOT_NODE_IDX], INUSE, AVAILABLE);
        }
        return;
    }

    if (threadId == MASTER_THREAD) {
        acquireLock(&heapLocks[targetNode], AVAILABLE, INUSE);
    }
    __syncthreads();

    copyArray(heap->arr, targetNode * BATCH_SIZE, heap->arr, ROOT_NODE_IDX * BATCH_SIZE, BATCH_SIZE);
    memsetArray(heap->arr, targetNode * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();

    if (threadId == MASTER_THREAD) {
        releaseLockAtomic(&heapLocks[targetNode], INUSE, AVAILABLE);
    }
    __syncthreads();

    copyArray(heap->arr, ROOT_NODE_IDX * BATCH_SIZE, array1, 0, BATCH_SIZE);
    memsetArray(heap->arr, ROOT_NODE_IDX * BATCH_SIZE, INT_MAX, BATCH_SIZE);
    __threadfence();

    copyArray(partialBuffer->arr, 0, array2, 0, partialBuffer->itemCount);
    __syncthreads();

    mergeAndSortArrays(array1, BATCH_SIZE, array2, partialBuffer->itemCount, mergedResult);

    copyArray(mergedResult, BATCH_SIZE, partialBuffer->arr, 0, partialBuffer->itemCount);
    __threadfence();

    copyArray(mergedResult, 0, array1, 0, BATCH_SIZE);

    int leftChild = 0, rightChild = 0, currentNode = 1;
    int largestLeftChild = 0, largestRightChild = 0;

    while (1) {
        if ((currentNode << 1) >= NUMBER_OF_NODES) {
            break;
        }

        leftChild = currentNode << 1;
        rightChild = leftChild + 1;

        if (threadId == MASTER_THREAD) {
            acquireLock(&heapLocks[leftChild], AVAILABLE, INUSE);
            acquireLock(&heapLocks[rightChild], AVAILABLE, INUSE);
        }
        __syncthreads();

        copyArray(heap->arr, leftChild * BATCH_SIZE, array2, 0, BATCH_SIZE);
        memsetArray(heap->arr, leftChild * BATCH_SIZE, INT_MAX, BATCH_SIZE);
        __threadfence();

        copyArray(heap->arr, rightChild * BATCH_SIZE, array3, 0, BATCH_SIZE);
        memsetArray(heap->arr, rightChild * BATCH_SIZE, INT_MAX, BATCH_SIZE);
        __threadfence();

        largestLeftChild = array2[BATCH_SIZE - 1];
        largestRightChild = array3[BATCH_SIZE - 1];

        mergeAndSortArrays(array2, BATCH_SIZE, array3, BATCH_SIZE, mergedResult);

        if (largestLeftChild > largestRightChild) {
            int temp = leftChild;
            leftChild = rightChild;
            rightChild = temp;
        }

        copyArray(mergedResult, BATCH_SIZE, heap->arr, rightChild * BATCH_SIZE, BATCH_SIZE);
        __threadfence();

        if (threadId == MASTER_THREAD) {
            releaseLockAtomic(&heapLocks[rightChild], INUSE, AVAILABLE);
        }
        __syncthreads();

        copyArray(mergedResult, 0, array2, 0, BATCH_SIZE);

        mergeAndSortArrays(array1, BATCH_SIZE, array2, BATCH_SIZE, mergedResult);

        copyArray(mergedResult, 0, heap->arr, currentNode * BATCH_SIZE, BATCH_SIZE);
        __threadfence();

        if (threadId == MASTER_THREAD) {
            releaseLockAtomic(&heapLocks[currentNode], INUSE, AVAILABLE);
        }
        __syncthreads();

        copyArray(mergedResult, BATCH_SIZE, array1, 0, BATCH_SIZE);
        currentNode = leftChild;
        __syncthreads();
    }

    __syncthreads();

    copyArray(array1, 0, heap->arr, currentNode * BATCH_SIZE, BATCH_SIZE);
    __threadfence();

    if (threadId == MASTER_THREAD) {
        releaseLockAtomic(&heapLocks[currentNode], INUSE, AVAILABLE);
    }
    __syncthreads();
}





__host__ void initializeHeap() {
    cudaMalloc(&d_partialBuffer, sizeof(Partial_Buffer));
    cudaMalloc(&d_Heap, sizeof(Heap)); 
    cudaMalloc((void**)&d_heapLock, (1 + NUMBER_OF_NODES) * sizeof(int)) ;

    cudaMemsetAsync(d_heapLock, AVAILABLE, (1 + NUMBER_OF_NODES) * sizeof(int) );
    initializeHeap<<<ceil(HEAP_CAPACITY / 1024), 1024, 0>>>(d_Heap, d_partialBuffer);

    cudaDeviceSynchronize();

}

// Helper function to insert keys into the heap.
__host__ void insertKeys(int* itemsToInsert, int itemsToInsertSize) {
    if (itemsToInsertSize < 0) {
        return;
    }
    
    int keysInsertionCount = BATCH_SIZE;
    int* d_mergedResult; // Device pointer for mergedResult
    int mergedResultSize = 2 * BATCH_SIZE; // Adjust size as required
    cudaMalloc((void**)&d_mergedResult, mergedResultSize * sizeof(int));

    for (int i = 0; i < itemsToInsertSize; i += BATCH_SIZE) {
        keysInsertionCount = min(itemsToInsertSize - i, BATCH_SIZE);
        topDownInsertion<<<1, BLOCK_SIZE>>>(itemsToInsert + i, keysInsertionCount,
                                     d_heapLock, d_partialBuffer, d_Heap, d_mergedResult);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_mergedResult);
}

__host__ void deleteKeys(int* deletedItems, int deletedItemsSize) {
    int arraySize = BATCH_SIZE;
    int mergedResultSize = 4 * BATCH_SIZE;  // Adjust as needed, considering operations in the kernel

    int* d_array1, *d_array2, *d_array3, *d_mergedResult;
    cudaMalloc((void**)&d_array1, arraySize * sizeof(int));
    cudaMalloc((void**)&d_array2, arraySize * sizeof(int));
    cudaMalloc((void**)&d_array3, arraySize * sizeof(int));
    cudaMalloc((void**)&d_mergedResult, mergedResultSize * sizeof(int));

    for (int i = 0; i < deletedItemsSize; i += BATCH_SIZE) {
        int currentBatchSize = min(deletedItemsSize - i, BATCH_SIZE);
        topDownDeletion<<<1, BLOCK_SIZE>>>(deletedItems + i, d_heapLock, d_partialBuffer, d_Heap, d_array1, d_array2, d_array3, d_mergedResult);
        cudaPeekAtLastError();
        cudaDeviceSynchronize();
    }

    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_array3);
    cudaFree(d_mergedResult);
}

// Helper function to insert and delete keys from the heap. This is the third benchmark mentioned by Dr. Izzat
// I acknowledge that using both insertKeys and deleteKeys in this way sort of defeats the purpose of the test
// as it was mentioned that the same method should perform both, through the use of an insertTopDownDeleteTopDown method
// for example, but I was not able to implement such a complex method due to time constraints (although the extensions
// have been generous in this regard). I hope this is acceptable.

__host__ void insertAndDeleteKeys(int* itemsToInsert, int itemsToInsertSize, int* deletedItems, int deletedItemsSize, int interleavingFactor) {
    int keysInsertionCount = BATCH_SIZE;
    int insertCount = 0;
    int deleteCount = 0;

    int arraySize = BATCH_SIZE;
    int mergedResultSize = 4 * BATCH_SIZE;  // Make sure the size matches your requirement

    // Allocate memory for all arrays used in insertion and deletion
    int* d_array1, *d_array2, *d_array3, *d_mergedResultInsert, *d_mergedResultDelete;
    cudaMalloc((void**)&d_array1, arraySize * sizeof(int));
    cudaMalloc((void**)&d_array2, arraySize * sizeof(int));
    cudaMalloc((void**)&d_array3, arraySize * sizeof(int));
    cudaMalloc((void**)&d_mergedResultInsert, mergedResultSize * sizeof(int));
    cudaMalloc((void**)&d_mergedResultDelete, mergedResultSize * sizeof(int));

    while (insertCount < itemsToInsertSize || deleteCount < deletedItemsSize) {
        if (insertCount < interleavingFactor && insertCount < itemsToInsertSize) {
            keysInsertionCount = min(itemsToInsertSize - insertCount, BATCH_SIZE);
            topDownInsertion<<<1, BLOCK_SIZE>>>(itemsToInsert + insertCount, keysInsertionCount, d_heapLock, d_partialBuffer, d_Heap, d_mergedResultInsert);
            cudaDeviceSynchronize();
            insertCount += keysInsertionCount;
        } 
        if (deleteCount < deletedItemsSize) {
            topDownDeletion<<<1, BLOCK_SIZE>>>(deletedItems + deleteCount, d_heapLock, d_partialBuffer, d_Heap, d_array1, d_array2, d_array3, d_mergedResultDelete);
            cudaDeviceSynchronize();
            deleteCount++;
        }
    }

    // Free all allocated memory
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_array3);
    cudaFree(d_mergedResultInsert);
    cudaFree(d_mergedResultDelete);
}




__host__ void heap_finalise() {
    cudaFree(d_partialBuffer);
    cudaFree(d_Heap);
    cudaFree(d_heapLock);
}
