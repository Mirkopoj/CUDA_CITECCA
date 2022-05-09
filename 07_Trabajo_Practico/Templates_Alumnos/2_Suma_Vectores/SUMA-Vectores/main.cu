#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include "vector_io.h"
#include "vector_ops.h"


#ifndef N
#define N 100 //10000000
#endif

#ifndef VECES
#define VECES 1
#endif


int suma_secuencial(float *h_A, float *h_B, int size);
int suma_paralela(float *d_A, float *d_B, int size);

int main()
{
    /* detecto placa y su nombre */
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Computer name: %s \n ", deviceProp.name);


    /* TODO: aloque memoria en host para los vectores h_A, h_B, y h_aux*/
    /* alocacion de memoria en host */
    float *h_A, *h_B, *h_aux;
   // h_A = (float *) ...
   // h_B = (float *) ...
   // h_aux = (float *) ...  

    /* TODO: aloque memoria en device para d_A y d_B */
    /* alocacion de memoria en device */
    float *d_A, *d_B;
   // cudaMalloc(...); 
   // cudaMalloc(...);  

  
    /* chequeo de alocacion de memoria */
    if (!h_A || !h_B || !d_A || !d_B || !h_aux) {
        printf("Error alocando vectores \n");
        exit(-1);
    }

    /* inicializacion de vectores */
    printf("Inicializacion vector A \n");
    vector_io_initializeRND(h_A, N);
    printf("Inicializacion vector B \n");
    vector_io_initializeRND(h_B, N);

 
    /* TODO: resuleva la transferencia de datos cpu -> gpu (host -> device) */
    // cudaMemcpy(...); 
    // cudaMemcpy(...); 

    /* suma secuencial */ 
    printf("Suma secuencial (CPU)\n");
    suma_secuencial(h_A, h_B, N);
  
    /* suma paralela */
    printf("Suma paralela (GPU) \n");
    suma_paralela(d_A, d_B, N);


    /* TODO: resuleva la transferencia de datos desde GPU a CPU para testear la suma */
    /// cudaMemcpy(h_aux, d_A, ...);

    /* se chequea el ultimo resultado, despues de sumar VECES veces*/
    if(vector_ops_iguales(h_aux, h_A, N)) 
        printf("Test pasado! \n");
    else
        printf("Test no pasado! \n");


    /* TODO: complete para liberar memoria en host */      
    /* liberacion de memoria */
    // free(...);
    // free(...);
    // free(..);
    
    /* TODO: complete para liberar memoria en device */      
    /* liberacion de memoria en device*/
   // ...(d_A);
   // ...(d_B);

    return 0;
}



int suma_secuencial(float *h_A, float *h_B, int size)
{
    
    /* tomar el tiempo inicial */
    struct timeval start;
    gettimeofday(&start, NULL);

    
    int i;
    for(i = 0; i < VECES; i++)
    {
        vector_ops_suma_sec(h_A, h_B, size);
    }

    /* tomar el tiempo final */
    struct timeval finish;
    gettimeofday(&finish, NULL);

    /* imprimir el tiempo transcurrido */
    double time = ((finish.tv_sec - start.tv_sec) * 1000.0) + ((finish.tv_usec - start.tv_usec) / 1000.0);
    printf("Tiempo en CPU: %g ms \n", time);


    return 1;
}


int suma_paralela(float *d_A, float *d_B, int size)
{ 
   
    /* variables para tomar el tiempo en GPU: events */
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* tomar el tiempo inicial */
    cudaEventRecord(start,0);

    int i;
    for(i = 0; i < VECES; i++)
    {
        vector_ops_suma_par(d_A, d_B, size);
    }

    /* tomar el tiempo final y calcular tiempo transcurrido */ 
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Tiempo en GPU: %g ms \n", elapsedTime);

    return 1;
}


