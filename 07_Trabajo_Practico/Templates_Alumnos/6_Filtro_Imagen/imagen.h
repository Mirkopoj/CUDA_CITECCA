

/* entrada y salida de imagenes */
void leer_imagen(const char * nombre, float *imagen, int filas, int cols);
void salvar_imagen(const char * nombre, float *imagen, int filas, int cols);

/* inicializacion de filtros: se proponen 2 filtros, se pueden agregar más */
/* Estas inicializaciones son secuenciales, son pocos datos */
void inicializar_filtro_promedio(float *filtro, int tamFiltro);
void inicializar_filtro_enfocado(float *filtro, int tamFiltro);


/* Aplicacion de filtros secuenciales */
void filtro_sec_promedio(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro);
void filtro_sec_enfocado(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro);


/* Aplicacion de filtros paralelos */
void filtro_par_promedio(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro);
void filtro_par_enfocado(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro);

