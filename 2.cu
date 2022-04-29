#include <stdio.h>
#include <cuda.h>

__global__ void saludar(){
	printf("guenas\n");
}

int main(){
	printf("cuenas\n");
	saludar<<<1,10>>>();
	cudaDeviceReset();
	return 0;
}
