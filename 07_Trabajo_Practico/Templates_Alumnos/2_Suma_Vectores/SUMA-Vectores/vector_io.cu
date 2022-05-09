#include <stdio.h>
#include <time.h>
#include "vector_io.h"

/* inicializa el vector pasado como parametro con valores entre 0 y 99 */
int vector_io_initializeRND(float *v, int dim)
{
    srand(time(NULL));

    for (int i = 0; i < dim; i++) {
        v[i] = (float) (rand() % 100);
    }

    return 1;
}


/* imprime el vector pasado como parametro. */
int vector_io_print(float *v, int dim)
{
    for (int i = 0; i < dim; i++) {
        printf(" %.0f ", v[i]);
    }
    printf(" \n ");

    return 1;
}


/* inicializa el vector pasado como parametro con 1s */
int vector_io_initializeOnes(float *v, int dim)
{
    for (int i = 0; i < dim; i++) {
        v[i] = 1.0;
    }

    return 1;
}
