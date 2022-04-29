#include <stdio.h>
#include <cuda.h>

#define N 10

__global__ void suma(int* A, int* B, int* C){
	int i= threadIdx.x;
	C[i] = A[i] + B [i];
}

int main(){
	int c[N];
	{
	int a[N];
	int b[N];

	int* da;
	int* db;
	int* dc;

	for (int i = 0; i<N;i++){
		a[i]=i;
		b[i]=i*2;
		printf("%d\n",a[i]);
	}
		printf("\n");

	cudaMalloc((void**)&da, sizeof(int)*N);
	cudaMalloc((void**)&db, sizeof(int)*N);
	cudaMalloc((void**)&dc, sizeof(int)*N);
	
	cudaMemcpy(da, a, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, sizeof(int)*N, cudaMemcpyHostToDevice);
	}

	suma<<<1,N>>>(da,db,dc);
	cudaDeviceSynchronize();

	cudaMemcpy(c, dc, sizeof(int)*N, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N;i++){
		printf("%d\n",c[i]);
	}
	cudaReset()

	return 0;
}

