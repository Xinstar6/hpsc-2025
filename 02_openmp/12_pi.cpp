#include <cstdio>
#include <omp.h>

int main() {
  int n = 500000;
  double dx = 1. / n;
  double pi = 0;

  double start_time = omp_get_wtime();

  #pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }

  double end_time = omp_get_wtime();

  printf("%17.15f\n",pi);
  printf("Running Time : %f seconds\n", end_time - start_time);
}
