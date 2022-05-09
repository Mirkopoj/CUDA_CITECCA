#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_timer.h"
#include "cpu_timer.h"

/*
Suma de matrices
 */

#define FILAS 5000
#define COLS 5000

void inicilizar_Matriz(float *data, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        data[i] = (float)(rand() / 10.0f);
    }

    return;
}

void sum_matrix_sec(float *A, float *B, float *C, const int filas, const int cols)
{
    /* TODO: resolver la suma de matrices secuencial */
    // ...


}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f %d \n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Test pasado! \n\n");
    else
        printf("Test no pasado! \n\n");
}



// grid 2D block 2D
__global__ void sum_matrix_par(float *MatA, float *MatB, float *MatC, int cols, int filas)
{
    /* TODO: estudiar el indexado usando f y c */
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int f = threadIdx.y + blockIdx.y * blockDim.y;
    /* TODO: completar y asignar en idx el indice del thread */
    // int idx = ...;

    /* TODO: resuelva la suma de matrices */
    // if (c < cols && f < filas)  // por qué es necesario este control?
      //  MatC[idx] = ... ;
}






int main(int argc, char **argv)
{

    // datos de la placa
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Se usa placa %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);


    int nBytes = FILAS * COLS * sizeof(float);
    printf("Tamanio de Matrix : %d x  %d \n", FILAS, COLS);


    /* TODO: aloque memoria en host para las matrices h_A y h_B y las matrices resultados hostRef y gpuRef*/
      // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
   // h_A = ...
   // h_B = ...
   // hostRef = ...
   // gpuRef = ...


    // inicilizacion de matrice en host
    cpu_timer crono_cpu;
    crono_cpu.tic();
    inicilizar_Matriz(h_A, FILAS*COLS);
    inicilizar_Matriz(h_B, FILAS*COLS);
    crono_cpu.tac();
    printf("Inicializacion de matrices en host  %f msec\n", crono_cpu.elapsed());

    /* Inicializacion de las matrices resultado en 0*/
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    // add matrix at host side for result checks
    crono_cpu.tic();
    sum_matrix_sec(h_A, h_B, hostRef, FILAS, COLS);
    crono_cpu.tac();

    printf("Suma de matrices en host  %lf msec\n", crono_cpu.elapsed());


    /* TODO: aloque memoria en device para las matrices d_MatA, d_MatB y d_MatbC */
    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
   // ...
   // ...
   // ...


    /* TODO: complete la transferencia de memoria de las matrices inicializadas desde host a device  (h_A -> d_MatA y h_B -> d_MatB) */
    // transfer data from host to device
    // cudaMemcpy(...);
    // cudaMemcpy(...);


    /* TODO: complete para armar una grilla de bloques de 32x32 threads, con los bloques que hagan falta para solucionar el problema */
    /* por ejemplo: */
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    // sacar los 1s y completar como corresponda
    dim3 grid (1 , 1);


    /* Lanzamiento del kernel al kernel */
    gpu_timer crono_gpu;
    crono_gpu.tic();

    sum_matrix_par<<<grid, block>>>(d_MatA, d_MatB, d_MatC, FILAS, COLS);
    cudaDeviceSynchronize();

    crono_gpu.tac();
    printf("Suma de matrices en GPU  %lf msec\n", crono_gpu.ms_elapsed);


    cudaGetLastError();

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, FILAS*COLS);

    /* TODO: desaloque memoria de device */
    // ...;
    // ...;
    // ...;

    /* TODO: desaloque memoria de host */
    // ...;
    // ...;
    // ...;
    // ...;

    return (0);
}
