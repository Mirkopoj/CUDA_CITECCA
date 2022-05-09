#include <stdio.h>
#include <cuda.h>


/* Kernel */
__global__ void saludar() {

	printf("Hola Mundo desde GPU\n");
}


int main() {

	printf("Hola mundo! desde CPU \n");
	
	saludar <<<1,10>>>();
	cudaDeviceReset();

	return 0;
} 




