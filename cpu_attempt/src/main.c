#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 5) {
        printf("Usage: %s <input_file> <output_file> [levels nodeCapacity]\n", argv[0]);
        return 1;
    }

    FILE* inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        printf("Failed to open input file.\n");
        return 1;
    }

    FILE* outputFile = fopen(argv[2], "w");
    if (outputFile == NULL) {
        printf("Failed to open output file.\n");
        fclose(inputFile);
        return 1;
    }

    int levels = 20;
    int nodeCapacity = 100;
    if (argc == 5) {
        levels = atoi(argv[3]);
        nodeCapacity = atoi(argv[4]);
    }

    Heap* heap = createHeap(levels, nodeCapacity);

    clock_t start_insert, end_insert;
    double insert_time_used;
    start_insert = clock();

    int num;
    while (fscanf(inputFile, "%d", &num) == 1) {
        insertKey(heap, num);
    }

    end_insert = clock();
    insert_time_used = ((double) (end_insert - start_insert)) / CLOCKS_PER_SEC;
    printf("Insertion time: %f seconds\n", insert_time_used);

    clock_t start_deleteMin, end_deleteMin;
    double deleteMin_time_used;
    start_deleteMin = clock();

    int minKey;
    while ((minKey = deleteMin(heap)) != -1) {
        fprintf(outputFile, "%d\n", minKey);
    }

    end_deleteMin = clock();
    deleteMin_time_used = ((double) (end_deleteMin - start_deleteMin)) / CLOCKS_PER_SEC;
    printf("DeleteMin time: %f seconds\n", deleteMin_time_used);

    double total_time_used = insert_time_used + deleteMin_time_used;
    printf("Total time: %f seconds\n", total_time_used);

    fclose(inputFile);
    fclose(outputFile);
    freeHeap(heap);

    return 0;
}