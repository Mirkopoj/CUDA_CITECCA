#include <cstdio>
#include <cstdlib>
#include <cassert>


/* Size of the input data */
#define N 8388608  

/* Floating point type */
//typedef float FLOAT;
typedef double FLOAT;




/* kERNEL 1 */
__global__ void cuadrado(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  if (j < N) {

	  	FLOAT temp;
	
	  	temp = input[j] * input[j];
	  	output[j] = temp;	
	  }
}



/* KERNEL 2 */
__global__ void operaciones(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  
	  if (j < N) {
	  	FLOAT temp;
	
	  	temp = input[j] * input[j];
	  	temp = temp / input[j];
	  	temp = pow(temp,2);
	  	temp = sinf(temp);
	  	output[j] = temp;	
	  }
}


/* KERNEL 3 */
__global__ void operaciones_no_coalescido_divergencia(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  if (j < N) {
	  
	  	FLOAT temp;
	
	 	temp = input[(j*2) % N];
	  	temp = temp / 100;
	 
	 	if (j % 2)
	  		temp = sqrt(temp);
	  	else
	  		temp = cos(temp);
	  	temp = sin(temp);
	  	temp = pow(temp,2);
	  	temp = cos(temp);
	  	output[j] = temp;	
	  }
}


/* KERNEL 4 */
__global__ void operaciones_double_divergencia(FLOAT* input, FLOAT* output) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  if (j < N) {
	  
	  	double temp;
	
	 	temp = (double)input[j];
	  	temp = temp / 100;
	 
	 	if (j % 2)
	  		temp = sqrt(temp);
	  	else
	  		temp = cos(temp);
	  	temp = sin(temp);
	  	temp = pow(temp,2);
	  	temp = cos(temp);
	  	output[j] = (float)temp;	
	  }
}

////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
	cudaDeviceProp deviceProp;
    int dev; 

    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);


	/* Allocate memory on host */
	FLOAT *h_input = (FLOAT *) malloc(N * sizeof(FLOAT));  /* Input data */
	FLOAT *h_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* Output data */



	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) 
		h_input[i] = (FLOAT)(rand() % 100); 
	

	/* Allocate memory on device */
	FLOAT *d_input, *d_output;
	cudaMalloc((void**)&d_input, N * sizeof(FLOAT));
	cudaMalloc((void**)&d_output, N * sizeof(FLOAT));
	
	/* Copy input array to device */
	cudaMemcpy(d_input, h_input, N * sizeof(FLOAT), cudaMemcpyHostToDevice);

	
	
	dim3 block_size(512);
  	dim3 grid_size((N + block_size.x- 1) / block_size.x);

	cuadrado<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();
	

	operaciones<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();

	operaciones_no_coalescido_divergencia<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();

	operaciones_double_divergencia<<<grid_size, block_size>>>(d_input, d_output);
	cudaDeviceSynchronize();

	/* Free memory on host */
	free(h_input);
	free(h_output);


	/* Free memory on device */
	cudaFree(d_input);
	cudaFree(d_output);

	
}

