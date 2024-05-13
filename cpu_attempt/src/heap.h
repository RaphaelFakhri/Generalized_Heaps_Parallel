#ifndef HEAP_H
#define HEAP_H

typedef struct {
    int* keys;
    int size;
    int capacity;
} HeapNode;

typedef struct {
    HeapNode** nodes;
    int* partialBuffer;
    int partialBufferSize;
    int partialBufferCapacity;
    int levels;
    int nodeCapacity;
} Heap;

Heap* createHeap(int levels, int nodeCapacity);
void insertKey(Heap* heap, int key);
int deleteMin(Heap* heap);
void heapifyDown(Heap* heap, int level, int index);
int mergeAndSort(int* keys1, int size1, int* keys2, int size2, int* result);
int compareInts(const void* a, const void* b);
void freeHeap(Heap* heap);

#endif