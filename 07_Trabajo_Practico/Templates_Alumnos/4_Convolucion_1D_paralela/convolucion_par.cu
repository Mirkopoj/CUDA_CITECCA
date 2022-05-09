#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "gpu_timer.h"
#include "cpu_timer.h"



/* Tamanio del array de input */
const int N = 512*2500; 

/* Tamanio del filtro (debe ser divisor de N) */
const int M = 32*4;

/* Floating point type */
typedef float FLOAT;
//typedef double FLOAT;


/* Funcion para preparar el filtro */
void SetupFilter(FLOAT* filter, size_t size) 
{
	/* TODO: llene el filtro */
	for(int i = 0; i < size; i++) {
	//	filter[i] = ... ;// llene con cualquier cosa que le interese...
	}
}


/* Convolucion en la cpu */
void conv_cpu(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{
	/* Ayuda: se implementa la convolucion secuencial. 
	   Tenga en cuenta que esta es una posible solución muy simple al problema  */
	FLOAT temp;
	
	/*Barrido del vector input (tamaño N) y para cada elemento j hasta N hago la operacion*/
	/*de convolucion: elemento i del vector filter por elemento i+j del vector input */
	for(int j = 0; j < N; j++){	
		temp = 0.0;
		for(int i = 0; i < M; i++){
			temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}

}


/* convolucion usando indexado unidimensional de threads/blocks
 un thread por cada elemento del output todo en memoria global*/
__global__ void conv_gpu (const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	
	/* TODO: implemente la convolucion paralela  */

	//int j = ...;
	
	/*Barro vector input (tamaño N) y para cada elemento j hasta N hago la operacion*/
	/*de convolucion: elemento i del vector filter por elemento i+j del vector input */
//	output[j]=  ...;
	for(int i = 0; i < M; i++){
	//  	output[j] += ...;;
	}	
}



int main() 
{

	/* Imprime input/output general */
	printf("Input size N: %d\n", N);
	printf("Filter size M: %d\n", M);

	/* se imprime el nobre de la placa */
	int card;
	cudaGetDevice(&card);
    	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, card);
	printf("\nDevice %d: \"%s\" \n", card, deviceProp.name);


	/* chequeo que las dimensiones N y M sean correctas para esta solucion*/
	assert((N % M == 0) && (M < 1024));

	/* TODO: aloque memoria en el host para la senial, filtro, salida y salida para comparar resultados con gpu*/
	/* Aloca Memoria en el HOST para el input, el output, y el "check output" */
	FLOAT *h_input, *h_output, *check_output, *h_filter;
	//h_input = ...;  		/* Vector Input -> N  con padding tamanio M para evitar overfloat*/
	//h_output = ...; 		/* Vector Output -> N  del calculo en GPU */
	//check_output = ...; 		/* Check-Output -> N  del calculo en CPU*/
	//h_filter = ...;		/* Vector filtro ->M */



	/* Inicializa el filtro */
	 SetupFilter(h_filter, M);

	/* Llena el array de input (CON "padding") con numeros aleatorios acotados */
	for(int i = 0 ; i < N+M ; i++){
		h_input[i] = (FLOAT)(rand() % 100); 
	}


	/* TODO: aloque memoria en device para la senial, filtro y la salida */
	FLOAT *d_input, *d_output, *d_filter;  // recordar que d_input tiene padding
	//cuda...
        //cuda...
	//cuda...
    
	// setear a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));


	/* TODO: copiar en device la senial de entrada y el filtro*/
//	cuda...
//	cuda...
	
	/* cronometraje */
	cpu_timer crono_cpu; 
	crono_cpu.tic();

	/* check en la CPU */
	conv_cpu(h_input, check_output, h_filter);

	crono_cpu.tac();


  	/*Defino tamaño bloque y grilla. Esta solucion es para M < 1024 y M multiplo de N */
  	dim3 block_size(M);
  	dim3 grid_size(N/M);

  	gpu_timer crono_gpu;
  	crono_gpu.tic();
    
    	/* TODO: realice el lanzamiento del kernel, con la grilla de grid_size bloques y block_size theads por bloques*/
    	//...

	crono_gpu.tac();

	/* TODO: copiar el resultado de device a host usando d_output y h_output  */
	//cudaMemcpy( h_output, d_output, ... , ... );


	/* Comparacion (lea documentacion de la funcion de C assert si no la conoce)*/	
	for(int j=0; j<N; j++){
		assert(h_output[j] == check_output[j]);
	}


   /* Impresion de tiempos */
	printf("[N/M/ms_cpu/ms_gpu/ms_gpu]= [%d/%d/%lf/%lf] \n", N, M, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed);


	/* TODO: libere memoria en host y device */
	// ...
	// ... ...

}

