#ifndef BINARY_HEAP_H
#define BINARY_HEAP_H

#include <vector>

class BinaryHeap {
private:
    std::vector<int> heap;

    void sift_up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (heap[parent] > heap[index]) {
                std::swap(heap[parent], heap[index]);
                index = parent;
            } else {
                break;
            }
        }
    }

    void siftDown(int index) {
        int n = heap.size();
        while (2 * index + 1 < n) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int smallest = index;
            if (leftChild < n && heap[leftChild] < heap[smallest]) {
                smallest = leftChild;
            }
            if (rightChild < n && heap[rightChild] < heap[smallest]) {
                smallest = rightChild;
            }
            if (smallest != index) {
                std::swap(heap[index], heap[smallest]);
                index = smallest;
            } else {
                break;
            }
        }
    }

public:
    BinaryHeap(int capacity) {
        heap.reserve(capacity);
    }

    void insert(int value) {
        heap.push_back(value);
        sift_up(heap.size() - 1);
    }

    int deleteMin() {
        int minValue = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        siftDown(0);
        return minValue;
    }

    bool isEmpty() {
        return heap.empty();
    }
};

#endif