#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

// make all values in buckets be 0
__global__ void clearBuckets(int *buckets, int howManyBuckets) {
    int id = threadIdx.x;  // each thread gets one bucket to clear
    if (id < howManyBuckets) {
        buckets[id] = 0; 
    }
}

// each thread checks a number and puts it in the right bucket
__global__ void countNumbers(const int *numbers, int *buckets, int totalNumbers) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // global thread ID
    if (id < totalNumbers) {
        int number = numbers[id]; // read the number
        atomicAdd(&buckets[number], 1); 
    }
}

// write sorted numbers back into final result
__global__ void writeSortedNumbers(int *result, const int *buckets, const int *startingPoints, int howManyBuckets) {
    int id = threadIdx.x;  // each thread takes care of one bucket
    if (id < howManyBuckets) {
        int howMany = buckets[id];    // how many of this number we have
        int startAt = startingPoints[id];  // where we should start writing
        for (int j = 0; j < howMany; ++j) {
            result[startAt + j] = id;  // write the number again and again
        }
    }
}

int main() {
    int n = 50;  
    int range = 5;   

    // use same memory that both CPU and GPU can access
    int *key, *buckets, *sortedResult, *writeStarts;
    cudaMallocManaged(&key, n * sizeof(int));
    cudaMallocManaged(&buckets, range * sizeof(int));
    cudaMallocManaged(&sortedResult, n * sizeof(int));
    cudaMallocManaged(&writeStarts, range * sizeof(int));

    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    // clear all the buckets
    clearBuckets<<<1, range>>>(buckets, range);
    cudaDeviceSynchronize();

    // count how many times each number appears
    int threadsPerBlock = 256;
    int totalBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    countNumbers<<<totalBlocks, threadsPerBlock>>>(key, buckets, n);
    cudaDeviceSynchronize();

    // find out where to start to write number in final list 
    writeStarts[0] = 0;
    for (int i = 1; i < range; ++i) {
        writeStarts[i] = writeStarts[i - 1] + buckets[i - 1];
    }

    writeSortedNumbers<<<1, range>>>(sortedResult, buckets, writeStarts, range);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; ++i) {
        printf("%d ", sortedResult[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(buckets);
    cudaFree(sortedResult);
    cudaFree(writeStarts);

    return 0;
}
