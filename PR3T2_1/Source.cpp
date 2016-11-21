#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <mkl.h>

int N;
#define NTEST 100
int NV[13] = {100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000};

using namespace std;

void tarea2_1();

int main(int argc, char *argv[]){
	for (int i = 0; i < 13; i++) {
		N = NV[i];
		printf("N= %d\n", N);
		tarea2_1();
	}
	std::getchar();
	return 0;
}

void tarea2_1()
{
	double inicio, fin = dsecnd();
	double *A = (double*)mkl_malloc(N*N * sizeof(double), 64);
	double *B = (double*)mkl_malloc(N * sizeof(double), 64);
	int *pivot = (int*)mkl_malloc(N * sizeof(int), 32);

	std::default_random_engine generador;
	std::normal_distribution<double> aleatorio(0, 1);
	for (long i = 0; i < N; i++) {
		A[i] = aleatorio(generador);
		B[i] = aleatorio(generador);
	}
	for (long i = N; i < N*N; i++) {
		A[i] = aleatorio(generador);
	}
	for (int i = 0; i < N; i++) {
		A[i*N + 1] += 10;
	}

	int result;
	inicio = dsecnd();
	for (int i = 0; i < NTEST; i++) {
		result = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, A, N, pivot, B, 1);
		if (result != 0){
			fprintf(stderr, "Fallo en dgesv: %d\n", result);
			return;
		}
	}
	fin = dsecnd();
	double tiempo = (fin - inicio) / NTEST;
	printf("Tiempo: %lf msec\n\n", tiempo*1e3);
	mkl_free(A);
	mkl_free(B);
}