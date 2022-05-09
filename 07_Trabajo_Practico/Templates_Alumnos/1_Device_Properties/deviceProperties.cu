#include <stdio.h>
#include <cuda_runtime_api.h>

int main() {


	int count = 0;

	/*  cudaGetDeviceCount no retorna cudaSuccess */
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		printf("CUDA failed \n");
		exit (-1);
	}

	/*  cudaGetDeviceCount retorna 0 */
	if (count == 0) {
		printf("No hay placa que soporte CUDA \n");
		exit (0);
	}


	/*  cudaGetDeviceCount retorna cantidad de placas instaladas */
	int dev;
	cudaDeviceProp prop;


	/* se imprimen los datos de cada placa instalada */
	for (dev = 0; dev < count; dev++) {

		if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
			printf("Error \n");
			exit (-1);
		}

		printf( "\n" );
		printf( " --- General Information for device %d ---\n", dev );
		printf("Compute name: %s \n", prop.name);
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );

		printf( " --- Memory Information for device %d ---\n", dev );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );

		printf( " --- MP Information for device %d ---\n", dev );
		printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d)\n",	prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

		printf( "\n" );
	}


	return 0;
}
