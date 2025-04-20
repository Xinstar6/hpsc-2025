#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  double start_time = omp_get_wtime();
  int n = 500000;
  int range = 500000;
  
  std::vector<int> key(n);
  // generate random values
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  //printf("\n");

  std::vector<int> bucket(range,0); 
  // count the number of times each key value appears
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
      #pragma omp atomic
      bucket[key[i]]++;
  }

  std::vector<int> offset(range,0);

  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];

  std::vector<int> final_key(n);

  #pragma omp parallel for
  for (int i = 0; i < range; i++) {
      int start = offset[i];
      for (int j = 0; j < bucket[i]; j++) {
          final_key[start + j] = i;
      }
  }

  key = final_key;
  double end_time = omp_get_wtime();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  printf("Running Time : %f seconds\n", end_time - start_time);
}
