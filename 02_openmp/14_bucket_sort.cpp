#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  // generate random values
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  // count the number of times each key value appears
  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  std::vector<int> offset(range,0);

  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];

  #pragma omp parallel
  {
      // each thread gets its own private bucket
      std::vector<int> private_bucket(range, 0); 

      // each thread store local count
      #pragma omp for nowait
      for (int i = 0; i < n; i++) {
          private_bucket[key[i]]++;
      }

      // add the local to global count 
      #pragma omp critical 
      {
          for (int i = 0; i < range; i++) {
              bucket[i] += private_bucket[i];
          }
      }
  }
  // update the offset value
  for (int i = 1; i < range; i++) {
      offset[i] = offset[i - 1] + bucket[i - 1];
  }

  std::vector<int> final_key(n);

  #pragma omp parallel for
  for (int i = 0; i < range; i++) {
      int start = offset[i];
      for (int j = 0; j < bucket[i]; j++) {
          final_key[start + j] = i;
      }
  }

  key = final_key;

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
