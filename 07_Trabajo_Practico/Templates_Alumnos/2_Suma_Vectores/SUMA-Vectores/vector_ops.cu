#include "vector_ops.h"
#include <stdio.h>

__global__ void kernel_suma(float *v1, float *v2, int dim);

/* operaciones con vectores - implementacion */


/* Suma de vectores (inplace)  */
int vector_ops_suma_sec(float *v1, float *v2, int dim)
{

    /* TODO: resuelva la suma secuencial suma inplace (el resultado queda en el paramatro v1*/
   // for (...) {
   //     v1[i] = ...;
   // }

    return 1;
}





/* Suma de vectores. Resultado queda en el primer argumento */
int vector_ops_suma_par(float *v1, float *v2, int dim)
{

    /* TODO: configuracion de la grilla. Complete con valores validos para generar una grilla 1D 
             asumiendo que dim < cantidad de threads máximo en dimension x de la grilla */
   // dim3 nThreads(...);  
   // dim3 nBlocks(...);

    /* TODO: complete la invocacion del  kernel */
   // kernel_suma<<<... , ...>>>(v1, v2, dim);   

    cudaDeviceSynchronize();   

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("kernel error \n");
        exit(-1);
    }

    return 1;
}




/* suma de cada elemento del vector */
__global__ void kernel_suma(float *v1, float *v2, int dim)
{

    /* TODO: setee id con el identificador del tread dentro del bloque y relativo a toda la grilla */
    //int id = ... ; 

    /* TODO: descomente */
    // if (id < dim)
    //{
        /* TODO: resuelva la suma del elemento i del vector V1 (sumando V1 = V1 + V2 -> inplace) */
        // v1[id] = ...;
    //}
}




/* retorna 1 si los vectores son iguales, 0 cc */
int vector_ops_iguales(float *v1, float *v2, int dim)
{
    int i;
    for(i=0; i < dim; i++) {
        if(v1[i] != v2[i]) 
           return 0;
        
    }

    return 1;
}
