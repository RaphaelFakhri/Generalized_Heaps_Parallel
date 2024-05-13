#include "heap.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Heap* createHeap(int levels, int nodeCapacity) {
    Heap* heap = (Heap*)malloc(sizeof(Heap));
    heap->levels = levels;
    heap->nodeCapacity = nodeCapacity;
    heap->nodes = (HeapNode**)malloc(levels * sizeof(HeapNode*));
    for (int i = 0; i < levels; i++) {
        heap->nodes[i] = (HeapNode*)malloc((1 << i) * sizeof(HeapNode));
        for (int j = 0; j < (1 << i); j++) {
            heap->nodes[i][j].keys = (int*)malloc(nodeCapacity * sizeof(int));
            heap->nodes[i][j].size = 0;
            heap->nodes[i][j].capacity = nodeCapacity;
        }
    }

    heap->partialBuffer = (int*)malloc(nodeCapacity * sizeof(int));
    heap->partialBufferSize = 0;
    heap->partialBufferCapacity = nodeCapacity;

    return heap;
}

void insertKey(Heap* heap, int key) {
    // printf("Inserting key: %d\n", key);

    // Insert the key into the partial buffer
    heap->partialBuffer[heap->partialBufferSize++] = key;

    // Sort the partial buffer
    qsort(heap->partialBuffer, heap->partialBufferSize, sizeof(int), compareInts);

    // If the partial buffer is full, merge and sort with the root node
    if (heap->partialBufferSize == heap->partialBufferCapacity) {
        // printf("Partial buffer is full. Merging with root node.\n");

        int* mergedKeys = (int*)malloc((heap->nodeCapacity + heap->partialBufferSize) * sizeof(int));
        int mergedSize = mergeAndSort(heap->nodes[0][0].keys, heap->nodes[0][0].size, heap->partialBuffer, heap->partialBufferSize, mergedKeys);

        // Update the root node with the merged keys
        memcpy(heap->nodes[0][0].keys, mergedKeys, heap->nodeCapacity * sizeof(int));
        heap->nodes[0][0].size = (mergedSize < heap->nodeCapacity) ? mergedSize : heap->nodeCapacity;

        // Update the partial buffer with the remaining keys
        heap->partialBufferSize = (mergedSize > heap->nodeCapacity) ? (mergedSize - heap->nodeCapacity) : 0;
        memcpy(heap->partialBuffer, mergedKeys + heap->nodeCapacity, heap->partialBufferSize * sizeof(int));

        free(mergedKeys);

        // printf("Root node after merging:\n");
        // for (int i = 0; i < heap->nodes[0][0].size; i++) {
        //     printf("%d ", heap->nodes[0][0].keys[i]);
        // }
        // printf("\n");

        // If the root node is full, propagate the insertion down the heap
        if (heap->nodes[0][0].size == heap->nodeCapacity) {
            // printf("Root node is full. Propagating down the heap.\n");
            heapifyDown(heap, 0, 0);
        }
    }
}

int deleteMin(Heap* heap) {
    // printf("Deleting minimum key.\n");

    if (heap->partialBufferSize > 0) {
        int minKey = heap->partialBuffer[0];

        // Remove the minimum key from the partial buffer
        heap->partialBufferSize--;
        memmove(heap->partialBuffer, heap->partialBuffer + 1, heap->partialBufferSize * sizeof(int));

        // printf("Minimum key found in partial buffer: %d\n", minKey);

        if (heap->nodes[0][0].size > 0 && heap->partialBufferSize < heap->partialBufferCapacity) {
            heap->partialBuffer[heap->partialBufferSize++] = heap->nodes[0][0].keys[0];
            heap->nodes[0][0].size--;
            memmove(heap->nodes[0][0].keys, heap->nodes[0][0].keys + 1, heap->nodes[0][0].size * sizeof(int));
            heapifyDown(heap, 0, 0);
        }

        return minKey;
    }

    if (heap->nodes[0][0].size == 0) {
        // printf("Heap is empty.\n");
        return -1; // Heap is empty
    }

    int minKey = heap->nodes[0][0].keys[0];
    heap->nodes[0][0].size--;
    memmove(heap->nodes[0][0].keys, heap->nodes[0][0].keys + 1, heap->nodes[0][0].size * sizeof(int));

    // printf("Minimum key found in root node: %d\n", minKey);

    if (heap->nodes[0][0].size == 0) {
        // If the root node becomes empty, propagate the deletion up the heap
        // printf("Root node is empty. Propagating up the heap.\n");

        int level = 0;
        int index = 0;
        while (level < heap->levels - 1) {
            int leftChildIndex = index * 2;
            int rightChildIndex = leftChildIndex + 1;
            int nextLevel = level + 1;

            if (heap->nodes[nextLevel][leftChildIndex].size == 0 &&
                heap->nodes[nextLevel][rightChildIndex].size == 0) {
                break;
            }

            if (heap->nodes[nextLevel][rightChildIndex].size == 0 ||
                (heap->nodes[nextLevel][leftChildIndex].size > 0 &&
                 heap->nodes[nextLevel][leftChildIndex].keys[0] < heap->nodes[nextLevel][rightChildIndex].keys[0])) {
                heap->nodes[level][index] = heap->nodes[nextLevel][leftChildIndex];
                heap->nodes[nextLevel][leftChildIndex].size = 0;
                index = leftChildIndex;
            } else {
                heap->nodes[level][index] = heap->nodes[nextLevel][rightChildIndex];
                heap->nodes[nextLevel][rightChildIndex].size = 0;
                index = rightChildIndex;
            }

            // Sort the keys in the node
            qsort(heap->nodes[level][index].keys, heap->nodes[level][index].size, sizeof(int), compareInts);

            level = nextLevel;
        }
    } else {
        heapifyDown(heap, 0, 0);
    }

    return minKey;
}

void heapifyDown(Heap* heap, int level, int index) {
    // printf("Heapifying down at level %d, index %d\n", level, index);

    int leftChildLevel = level + 1;
    int leftChildIndex = index * 2;
    int rightChildIndex = leftChildIndex + 1;

    if (leftChildLevel >= heap->levels) {
        return;
    }

    int smallestIndex = index;
    int smallestLevel = level;

    if (leftChildIndex < (1 << leftChildLevel) && heap->nodes[leftChildLevel][leftChildIndex].size > 0 &&
        (heap->nodes[smallestLevel][smallestIndex].size == 0 ||
         heap->nodes[leftChildLevel][leftChildIndex].keys[0] < heap->nodes[smallestLevel][smallestIndex].keys[0])) {
        smallestIndex = leftChildIndex;
        smallestLevel = leftChildLevel;
    }

    if (rightChildIndex < (1 << leftChildLevel) && heap->nodes[leftChildLevel][rightChildIndex].size > 0 &&
        (heap->nodes[smallestLevel][smallestIndex].size == 0 ||
         heap->nodes[leftChildLevel][rightChildIndex].keys[0] < heap->nodes[smallestLevel][smallestIndex].keys[0])) {
        smallestIndex = rightChildIndex;
        smallestLevel = leftChildLevel;
    }

    if (smallestIndex != index) {
        // printf("Swapping keys at level %d, index %d with level %d, index %d\n", level, index, smallestLevel, smallestIndex);

        int temp = heap->nodes[level][index].keys[0];
        heap->nodes[level][index].keys[0] = heap->nodes[smallestLevel][smallestIndex].keys[0];
        heap->nodes[smallestLevel][smallestIndex].keys[0] = temp;

        // Sort the keys in the nodes
        qsort(heap->nodes[level][index].keys, heap->nodes[level][index].size, sizeof(int), compareInts);
        qsort(heap->nodes[smallestLevel][smallestIndex].keys, heap->nodes[smallestLevel][smallestIndex].size, sizeof(int), compareInts);

        heapifyDown(heap, smallestLevel, smallestIndex);
    }
}

int mergeAndSort(int* keys1, int size1, int* keys2, int size2, int* result) {
    int i = 0, j = 0, k = 0;
    while (i < size1 && j < size2) {
        if (keys1[i] <= keys2[j]) {
            result[k++] = keys1[i++];
        } else {
            result[k++] = keys2[j++];
        }
    }

    while (i < size1) {
        result[k++] = keys1[i++];
    }

    while (j < size2) {
        result[k++] = keys2[j++];
    }

    return k;
}

int compareInts(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

void freeHeap(Heap* heap) {
    for (int i = 0; i < heap->levels; i++) {
        for (int j = 0; j < (1 << i); j++) {
            free(heap->nodes[i][j].keys);
        }
        free(heap->nodes[i]);
    }
    free(heap->nodes);
    free(heap->partialBuffer);
    free(heap);
}